import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional
import os
from datetime import datetime

import numpy as np
import pandas as pd
import rtdl_num_embeddings
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # CPU 版 PyTorch 随机种子
    torch.manual_seed(seed)
    # GPU 版 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 强制 cuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


TaskType = Literal['regression', 'binclass', 'multiclass']
task_type: TaskType = 'regression'
n_classes = None

file_path = 'models/subgoal_dataset_0.35.txt'
target_cols = ['subgoal_x', 'subgoal_y']

if os.path.exists(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    # feature_cols = df.drop(columns=target_cols).columns.tolist()
    # print(f"Training with {len(feature_cols)} features: {feature_cols[:5]} ...")

    X_num = df.drop(columns=target_cols).values.astype(np.float32)
    Y = df[target_cols].values.astype(np.float32)
else:
    print("Warning: File not found, generating simulation data...")
    N_SAMPLES = 2000
    X_num = np.random.randn(N_SAMPLES, 8).astype(np.float32)
    y1 = X_num[:, 0] + X_num[:, 1] * 0.5
    y2 = X_num[:, 0] * -1 + X_num[:, 2]
    Y = np.column_stack([y1, y2]).astype(np.float32)

task_is_regression = task_type == 'regression'
n_num_features = X_num.shape[1]
n_outputs = Y.shape[1]

print(f"Input Features: {n_num_features}, Output Targets: {n_outputs}")

cat_cardinalities = []
X_cat = None

all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.9, random_state=42
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=8/9, random_state=42
)

data_numpy = {
    'train': {'x_num': X_num[train_idx], 'y': Y[train_idx]},
    'val': {'x_num': X_num[val_idx], 'y': Y[val_idx]},
    'test': {'x_num': X_num[test_idx], 'y': Y[test_idx]},
}

x_num_train_numpy = data_numpy['train']['x_num']
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, x_num_train_numpy.shape)
    .astype(x_num_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train_numpy + noise)
del x_num_train_numpy

for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])
    # 移除 nan_to_num 的 posinf/neginf 限制以及 np.clip，以避免数据截断
    data_numpy[part]['x_num'] = np.nan_to_num(data_numpy[part]['x_num'], nan=0.0)

class RegressionLabelStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray

Y_train = data_numpy['train']['y'].copy()
if task_type == 'regression':
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std[y_std == 0] = 1.0
    regression_label_stats = RegressionLabelStats(y_mean, y_std)
    Y_train = (Y_train - regression_label_stats.mean) / regression_label_stats.std
else:
    regression_label_stats = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(Y_train, device=device)
if task_type == 'regression':
    for part in data:
        data[part]['y'] = data[part]['y'].float()
    Y_train = Y_train.float()

amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
amp_enabled = True and amp_dtype is not None
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None

print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')

bins = rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=512)
num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    bins, # 将每个特征划分为512个区间
    d_embedding=32, # 每个特征映射到32维空间
    activation=False,
    version='B',
)

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=n_outputs,
    num_embeddings=num_embeddings,
    n_blocks=3, # 模型中残差块的数量
    d_block=640, # 每个块中隐藏层的维度（即神经元的数量）
    dropout=0.0,
    k=8,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-5)
gradient_clipping_norm: Optional[float] = 0.8
share_training_batches = True

@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
def apply_model(part: str, idx: Tensor) -> Tensor:
    out = model(
        data[part]['x_num'][idx],
        data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
    )
    if n_outputs == 1:
        out = out.squeeze(-1)
    return out.float()

base_loss_fn = lambda y_pred, y_true: nn.functional.huber_loss(
    y_pred, y_true, delta=1.0
)

def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    if share_training_batches:
        y_true = y_true.repeat_interleave(model.backbone.k, dim=0)
    else:
        y_true = y_true.flatten(0, 1)
    return base_loss_fn(y_pred, y_true)

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

    y_pred = y_pred.mean(axis=1)
    y_true = data[part]['y'].cpu().numpy()

    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    score = -(mse ** 0.5)

    return {'score': float(score), 'mse': mse, 'r2': r2}

print(f'Test score before training: {evaluate("test")["score"]:.4f}')

n_epochs = 180
train_size = len(train_idx)
batch_size = 256

# 余弦退火 + 线性 warmup 调度
warmup_epochs = min(10, max(1, n_epochs // 5))
_denom = max(1, n_epochs - warmup_epochs)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda epoch: (epoch + 1) / warmup_epochs
    if epoch < warmup_epochs
    else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / _denom)),
)

# TensorBoard writer - 使用时间戳区分每次运行，保留历史曲线
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(log_dir=f'runs/tabm_{run_timestamp}')
model_save_dir = 'models'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f'subgoal_tabm_{run_timestamp}.pt')

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

print("\nStarting Training...")
for epoch in range(n_epochs):
    if share_training_batches:
        batches = torch.randperm(train_size, device=device).split(batch_size)
    else:
        batches = (
            torch.rand((train_size, model.backbone.k), device=device)
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )

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
    # 也在每个 epoch 评估训练集并记录到 TensorBoard（与 val/test 保持一致）
    eval_train = evaluate('train')
    metrics = {'train': eval_train, 'val': eval_val, 'test': eval_test}
    val_score = eval_val['score']
    val_score_improved = val_score > best_checkpoint['metrics']['val']['score']

    # 按 epoch 调度学习率
    lr_scheduler.step()

    # Log scalars to TensorBoard
    if batch_count > 0:
        avg_train_loss = total_loss / batch_count
        writer.add_scalar('train/loss', avg_train_loss, epoch)
    writer.add_scalar('val/mse', eval_val['mse'], epoch)
    writer.add_scalar('val/r2', eval_val['r2'], epoch)
    writer.add_scalar('val/rmse', -eval_val['score'], epoch)
    writer.add_scalar('test/mse', eval_test['mse'], epoch)
    writer.add_scalar('test/r2', eval_test['r2'], epoch)
    writer.add_scalar('test/rmse', -eval_test['score'], epoch)
    # 记录训练集的 MSE 和 R²
    writer.add_scalar('train/mse', eval_train['mse'], epoch)
    writer.add_scalar('train/r2', eval_train['r2'], epoch)
    writer.add_scalar('train/rmse', -eval_train['score'], epoch)
    # log learning rate (first param group)
    try:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', lr, epoch)
    except Exception:
        pass

    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
        mark = "*"
    else:
        remaining_patience -= 1
        mark = " "

    print(
        f'{mark} [Epoch {epoch:03d}] '
        f'Train MSE: {eval_train["mse"]:.4f} | R2: {eval_train["r2"]:.4f}  '
        f'Val MSE: {eval_val["mse"]:.4f} | R2: {eval_val["r2"]:.4f}  '
        f'Test MSE: {eval_test["mse"]:.4f} | R2: {eval_test["r2"]:.4f}'
    )

    if remaining_patience < 0:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_checkpoint['model'])
final_res = best_checkpoint['metrics']['test']

print('\n' + '='*40)
print('Tabm RESULTS')
print('='*40)
print(f'MSE : {final_res["mse"]:.6f}')
print(f'R²  : {final_res["r2"]:.6f}')
print(f'RMSE: {(-final_res["score"]):.6f}')

torch.save(
    {
        'model_state_dict': best_checkpoint['model'],
        'optimizer_state_dict': best_checkpoint['optimizer'],
        'preprocessing': preprocessing,
        'regression_label_stats': regression_label_stats,
        'task_type': task_type,
        'timestamp': run_timestamp,
        'model_params': {
            'n_num_features': n_num_features,
            'n_outputs': n_outputs,
            'cat_cardinalities': cat_cardinalities,
            'bins': bins,
        },
    },
    model_save_path,
)
print(f'Best model checkpoint saved to {model_save_path}')

# Close TensorBoard writer
writer.close()
