import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional
import os
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TaskType = Literal['regression', 'binclass', 'multiclass']
task_type: TaskType = 'regression'

file_path = 'models/subgoal_dataset_4.txt'
target_cols = ['subgoal_x', 'subgoal_y']

if os.path.exists(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    X_num = df.drop(columns=target_cols).values.astype(np.float32)
    Y = df[target_cols].values.astype(np.float32)
else:
    print("Warning: File not found, generating simulation data...")
    # placeholder for simulation data if missing, normally shouldn't hit this based on your env
    X_num = np.random.randn(1000, 20).astype(np.float32)
    Y = np.random.randn(1000, 2).astype(np.float32)

n_num_features = X_num.shape[1]
n_outputs = Y.shape[1]

print(f"Input Features: {n_num_features}, Output Targets: {n_outputs}")

# 统一维护想要测试的多个 seed
SEEDS = [42, 100, 2026]

class RegressionLabelStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray

class MLP(nn.Module):
    def __init__(self, d_in, d_layers, d_out, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = d_in
        for d in d_layers:
            layers.append(nn.Linear(in_dim, d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = d
        layers.append(nn.Linear(in_dim, d_out))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
amp_enabled = True and amp_dtype is not None
print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')

all_idx = np.arange(len(Y))
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
final_metrics_list = []

# ======= 开始不同 Seed 的循环 =======
for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f'Starting Training for SEED: {seed}')
    print(f"{'='*60}")

    set_seed(seed)

    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.9, random_state=seed
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=8/9, random_state=seed
    )

    data_numpy = {
        'train': {'x_num': X_num[train_idx].copy(), 'y': Y[train_idx].copy()},
        'val': {'x_num': X_num[val_idx].copy(), 'y': Y[val_idx].copy()},
        'test': {'x_num': X_num[test_idx].copy(), 'y': Y[test_idx].copy()},
    }

    x_num_train_numpy = data_numpy['train']['x_num']
    noise = (
        np.random.default_rng(seed)
        .normal(0.0, 1e-5, x_num_train_numpy.shape)
        .astype(x_num_train_numpy.dtype)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
        output_distribution='normal',
        subsample=10**9,
        random_state=seed, 
    ).fit(x_num_train_numpy + noise)
    
    for part in data_numpy:
        data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])
        data_numpy[part]['x_num'] = np.nan_to_num(data_numpy[part]['x_num'], nan=0.0)

    Y_train = data_numpy['train']['y'].copy()
    if task_type == 'regression':
        y_mean = Y_train.mean(axis=0)
        y_std = Y_train.std(axis=0)
        y_std[y_std == 0] = 1.0
        regression_label_stats = RegressionLabelStats(y_mean, y_std)
        Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
    else:
        regression_label_stats = None

    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }
    Y_train = torch.as_tensor(Y_train, device=device)
    if task_type == 'regression':
        for part in data:
            data[part]['y'] = data[part]['y'].float()
        Y_train = Y_train.float()

    grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None

    # 类似于 tabm 中设置的维度，我们使用几个隐藏层
    model = MLP(d_in=n_num_features, d_layers=[512, 512, 512], d_out=n_outputs, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-5)
    gradient_clipping_norm: Optional[float] = 0.8

    @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
    def apply_model(part: str, idx: Tensor) -> Tensor:
        out = model(data[part]['x_num'][idx])
        if n_outputs == 1:
            out = out.squeeze(-1)
        return out.float()

    def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
        return nn.functional.huber_loss(y_pred, y_true, delta=1.0)

    @torch.no_grad()
    def evaluate(part: str) -> dict:
        model.eval()
        eval_batch_size = 8096
        y_pred_list = []
        indices = torch.arange(len(data[part]['y']), device=device)
        for batch_idx in indices.split(eval_batch_size):
            batch_pred = apply_model(part, batch_idx)
            y_pred_list.append(batch_pred.cpu())

        y_pred = torch.cat(y_pred_list).numpy()

        if regression_label_stats is not None:
            y_pred = y_pred * regression_label_stats.std + regression_label_stats.mean

        y_true = data[part]['y'].cpu().numpy()

        mde = float(np.mean(np.linalg.norm(y_pred - y_true, axis=1)))
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        score = -(mse ** 0.5)

        return {'score': float(score), 'mse': mse, 'r2': r2, 'mde': mde}

    n_epochs = 180
    train_size = len(train_idx)
    batch_size = 256

    warmup_epochs = min(10, max(1, n_epochs // 5))
    _denom = max(1, n_epochs - warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (epoch + 1) / warmup_epochs
        if epoch < warmup_epochs
        else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / _denom)),
    )

    writer_path = f'runs/mlp_multi_seed_{run_timestamp}/seed_{seed}'
    writer = SummaryWriter(log_dir=writer_path)
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'subgoal_mlp_{run_timestamp}_seed{seed}.pt')

    metrics = {'val': {'score': -math.inf}, 'test': {'score': -math.inf}}

    def make_checkpoint() -> dict[str, Any]:
        return deepcopy({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': metrics,
        })

    best_checkpoint = make_checkpoint()
    patience = 40
    remaining_patience = patience

    print("Start Training ...")
    for epoch in range(n_epochs):
        batches = torch.randperm(train_size, device=device).split(batch_size)

        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_idx in batches:
            optimizer.zero_grad()
            loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
            
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()
                
            if gradient_clipping_norm is not None:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), gradient_clipping_norm
                )
                
            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

            total_loss += float(loss.detach().cpu().item())
            batch_count += 1

        eval_val = evaluate('val')
        eval_test = evaluate('test')
        eval_train = evaluate('train')
        metrics = {'train': eval_train, 'val': eval_val, 'test': eval_test}
        val_score = eval_val['score']
        val_score_improved = val_score > best_checkpoint['metrics']['val']['score']

        lr_scheduler.step()

        if batch_count > 0:
            avg_train_loss = total_loss / batch_count
            writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('val/mse', eval_val['mse'], epoch)
        writer.add_scalar('val/r2', eval_val['r2'], epoch)
        writer.add_scalar('val/rmse', -eval_val['score'], epoch)
        writer.add_scalar('val/mde', eval_val['mde'], epoch)
        writer.add_scalar('test/mse', eval_test['mse'], epoch)
        writer.add_scalar('test/r2', eval_test['r2'], epoch)
        writer.add_scalar('test/rmse', -eval_test['score'], epoch)
        writer.add_scalar('test/mde', eval_test['mde'], epoch)
        writer.add_scalar('train/mse', eval_train['mse'], epoch)
        writer.add_scalar('train/r2', eval_train['r2'], epoch)
        writer.add_scalar('train/rmse', -eval_train['score'], epoch)
        writer.add_scalar('train/mde', eval_train['mde'], epoch)
        
        try:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/lr', lr, epoch)
        except Exception:
            pass

        writer.flush()

        if val_score_improved:
            best_checkpoint = make_checkpoint()
            remaining_patience = patience
            mark = "*"
        else:
            remaining_patience -= 1
            mark = " "

        if epoch % 10 == 0 or remaining_patience < 0 or epoch == n_epochs - 1:
            print(
                f'{mark} [Epoch {epoch:03d}] '
                f'Train MSE: {eval_train["mse"]:.4f} | R2: {eval_train["r2"]:.4f}  '
                f'Val MSE: {eval_val["mse"]:.4f} | R2: {eval_val["r2"]:.4f} | MDE: {eval_val["mde"]:.4f}  '
                f'Test MSE: {eval_test["mse"]:.4f} | R2: {eval_test["r2"]:.4f} | MDE: {eval_test["mde"]:.4f}'
            )

        if remaining_patience < 0:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_checkpoint['model'])
    final_res = best_checkpoint['metrics']['test']

    print('\n' + '-'*40)
    print(f'Seed {seed} RESULTS:')
    print(f'MSE : {final_res["mse"]:.6f}')
    print(f'R²  : {final_res["r2"]:.6f}')
    print(f'RMSE: {(-final_res["score"]):.6f}')
    print(f'MDE : {final_res["mde"]:.6f}')
    print('-'*40)

    final_metrics_list.append({
        'seed': seed,
        'mse': final_res["mse"],
        'r2': final_res["r2"],
        'rmse': -final_res["score"],
        'mde': final_res["mde"]
    })

    torch.save(
        {
            'model_state_dict': best_checkpoint['model'],
            'optimizer_state_dict': best_checkpoint['optimizer'],
            'preprocessing': preprocessing,
            'regression_label_stats': regression_label_stats,
            'task_type': task_type,
            'timestamp': run_timestamp,
            'seed': seed,
            'model_params': {
                'n_num_features': n_num_features,
                'n_outputs': n_outputs,
            },
        },
        model_save_path,
    )
    writer.close()

# ======= 汇总所有 Seed 的结果 =======
print('\n' + '='*60)
print(f'FINAL AGGREGATED RESULTS ACROSS {len(SEEDS)} SEEDS')
print('='*60)

agg_mse = [r['mse'] for r in final_metrics_list]
agg_r2 = [r['r2'] for r in final_metrics_list]
agg_rmse = [r['rmse'] for r in final_metrics_list]
agg_mde = [r['mde'] for r in final_metrics_list]

print(f"MSE  : {np.mean(agg_mse):.6f} ± {np.std(agg_mse):.6f}")
print(f"R²   : {np.mean(agg_r2):.4f} ± {np.std(agg_r2):.4f}")
print(f"RMSE : {np.mean(agg_rmse):.6f} ± {np.std(agg_rmse):.6f}")
print(f"MDE  : {np.mean(agg_mde):.6f} ± {np.std(agg_mde):.6f}")
