import os
import random
from datetime import datetime
from typing import Literal, NamedTuple, Optional

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from catboost import CatBoostRegressor, Pool
import joblib
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

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
    N_SAMPLES = 2000
    X_num = np.random.randn(N_SAMPLES, 8).astype(np.float32)
    y1 = X_num[:, 0] + X_num[:, 1] * 0.5
    y2 = X_num[:, 0] * -1 + X_num[:, 2]
    Y = np.column_stack([y1, y2]).astype(np.float32)

n_num_features = X_num.shape[1]
n_outputs = Y.shape[1]
print(f"Input Features: {n_num_features}, Output Targets: {n_outputs}")

# 统一维护想要测试的多个 seed
SEEDS = [42, 100, 2026]

# 数据划分
all_idx = np.arange(len(Y))
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
final_metrics_list = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f'Starting CatBoost Training for SEED: {seed}')
    print(f"{'='*60}")

    set_seed(seed)

    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        all_idx, train_size=0.9, random_state=seed
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=8/9, random_state=seed
    )

    data_numpy = {
        'train': {'x_num': X_num[train_idx], 'y': Y[train_idx]},
        'val': {'x_num': X_num[val_idx], 'y': Y[val_idx]},
        'test': {'x_num': X_num[test_idx], 'y': Y[test_idx]},
    }

    # 特征预处理
    x_num_train_numpy = data_numpy['train']['x_num'].copy()
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

    # 标签标准化
    class RegressionLabelStats(NamedTuple):
        mean: np.ndarray
        std: np.ndarray

    Y_train = data_numpy['train']['y'].copy()
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0)
    y_std[y_std == 0] = 1.0
    regression_label_stats = RegressionLabelStats(y_mean, y_std)
    Y_train_normalized = (Y_train - regression_label_stats.mean) / regression_label_stats.std

    # TensorBoard writer
    writer_path = f'runs/catboost_multi_seed_{run_timestamp}/seed_{seed}'
    writer = SummaryWriter(log_dir=writer_path)

    # CatBoost 参数
    catboost_params = {
        'iterations': 180,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'random_seed': seed,
        'loss_function': 'MultiRMSE',
        'early_stopping_rounds': 50,
        'verbose': 100,
        'task_type': 'GPU' if os.environ.get('USE_GPU', '0') == '1' else 'CPU',
    }

    # 准备验证集
    Y_val_normalized = (data_numpy['val']['y'] - regression_label_stats.mean) / regression_label_stats.std

    def evaluate(part: str, model: CatBoostRegressor, ntree_end: Optional[int] = None) -> dict:
        """评估模型性能（与 TabM/XGBoost 对齐：mse/rmse/r2/mde）"""
        if ntree_end is None:
            y_pred_normalized = model.predict(data_numpy[part]['x_num'])
        else:
            y_pred_normalized = model.predict(data_numpy[part]['x_num'], ntree_end=int(ntree_end))
        y_pred_normalized = np.asarray(y_pred_normalized, dtype=np.float32)
        if y_pred_normalized.ndim == 1:
            y_pred_normalized = y_pred_normalized[:, None]

        y_pred = y_pred_normalized * regression_label_stats.std + regression_label_stats.mean
        y_true = data_numpy[part]['y']

        mde = float(np.mean(np.linalg.norm(y_pred - y_true, axis=1)))
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        rmse = mse ** 0.5
        return {'mse': mse, 'r2': r2, 'rmse': rmse, 'mde': mde}


    # 训练：2D 联合回归
    print(f"\nTraining model for 2D target (subgoal_x, subgoal_y) with seed {seed}...")
    model = CatBoostRegressor(**catboost_params)
    train_pool = Pool(data_numpy['train']['x_num'], Y_train_normalized)
    val_pool = Pool(data_numpy['val']['x_num'], Y_val_normalized)
    model.fit(train_pool, eval_set=val_pool)

    # 写入 RMSE 曲线（每一次 boosting iteration 都有）
    evals_result = model.get_evals_result() or {}
    train_curve = None
    val_curve = None
    if 'learn' in evals_result and 'MultiRMSE' in evals_result['learn']:
        train_curve = evals_result['learn']['MultiRMSE']
    if 'validation' in evals_result and 'MultiRMSE' in evals_result['validation']:
        val_curve = evals_result['validation']['MultiRMSE']

    if train_curve is not None:
        for step, rmse_train in enumerate(train_curve):
            writer.add_scalar('train/rmse', rmse_train, step)
            writer.add_scalar('train/mse', float(rmse_train) ** 2, step)
    if val_curve is not None:
        for step, rmse_val in enumerate(val_curve):
            writer.add_scalar('val/rmse', rmse_val, step)
            writer.add_scalar('val/mse', float(rmse_val) ** 2, step)

    # 周期性计算更“语义化”的指标（r2/mde），让三种算法的同名曲线可以对齐对比。
    metric_eval_every = int(os.environ.get('CATBOOST_METRIC_EVERY', '10'))
    if metric_eval_every > 0 and val_curve is not None:
        for step in range(0, len(val_curve), metric_eval_every):
            res_train = evaluate('train', model, ntree_end=step + 1)
            res_val = evaluate('val', model, ntree_end=step + 1)
            res_test = evaluate('test', model, ntree_end=step + 1)
            writer.add_scalar('train/r2', res_train['r2'], step)
            writer.add_scalar('train/mde', res_train['mde'], step)
            writer.add_scalar('train/mse', res_train['mse'], step)
            writer.add_scalar('train/rmse', res_train['rmse'], step)
            writer.add_scalar('val/r2', res_val['r2'], step)
            writer.add_scalar('val/mde', res_val['mde'], step)
            writer.add_scalar('val/mse', res_val['mse'], step)
            writer.add_scalar('val/rmse', res_val['rmse'], step)

            writer.add_scalar('test/r2', res_test['r2'], step)
            writer.add_scalar('test/mde', res_test['mde'], step)
            writer.add_scalar('test/mse', res_test['mse'], step)
            writer.add_scalar('test/rmse', res_test['rmse'], step)

    # 最终评估
    eval_train = evaluate('train', model)
    eval_val = evaluate('val', model)
    eval_test = evaluate('test', model)

    print('\n' + '-'*40)
    print(f'Seed {seed} RESULTS:')
    print(f'Train - MSE: {eval_train["mse"]:.6f} | R²: {eval_train["r2"]:.6f} | RMSE: {eval_train["rmse"]:.6f} | MDE: {eval_train["mde"]:.6f}')
    print(f'Val   - MSE: {eval_val["mse"]:.6f} | R²: {eval_val["r2"]:.6f} | RMSE: {eval_val["rmse"]:.6f} | MDE: {eval_val["mde"]:.6f}')
    print(f'Test  - MSE: {eval_test["mse"]:.6f} | R²: {eval_test["r2"]:.6f} | RMSE: {eval_test["rmse"]:.6f} | MDE: {eval_test["mde"]:.6f}')
    print('-'*40)

    final_metrics_list.append({
        'seed': seed,
        'mse': eval_test["mse"],
        'r2': eval_test["r2"],
        'rmse': eval_test["rmse"],
        'mde': eval_test["mde"]
    })

    # 将最终 test 指标也放到统一的 tag 下（test 曲线通常只有 1 个点）
    final_step = len(val_curve) - 1 if val_curve is not None and len(val_curve) > 0 else 0
    for part, res in [('train', eval_train), ('val', eval_val), ('test', eval_test)]:
        writer.add_scalar(f'{part}/mse', res['mse'], final_step)
        writer.add_scalar(f'{part}/rmse', res['rmse'], final_step)
        writer.add_scalar(f'{part}/r2', res['r2'], final_step)
        writer.add_scalar(f'{part}/mde', res['mde'], final_step)

    # 保存模型
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'subgoal_catboost_{run_timestamp}_seed{seed}.joblib')

    joblib.dump(
        {
            'model': model,
            'models': [model],
            'preprocessing': preprocessing,
            'regression_label_stats': regression_label_stats,
            'task_type': task_type,
            'timestamp': run_timestamp,
            'seed': seed,
            'metrics': {'train': eval_train, 'val': eval_val, 'test': eval_test},
            'catboost_params': catboost_params,
        },
        model_save_path,
    )
    print(f'\nModel saved to {model_save_path}')

    # 记录最终指标到 TensorBoard
    writer.add_hparams(
        {'model': 'CatBoost', 'iterations': catboost_params['iterations'], 
         'depth': catboost_params['depth'], 'learning_rate': catboost_params['learning_rate'], 'seed': seed},
        {
            'final/train_mse': eval_train['mse'],
            'final/train_r2': eval_train['r2'],
            'final/train_rmse': eval_train['rmse'],
            'final/train_mde': eval_train['mde'],
            'final/val_mse': eval_val['mse'],
            'final/val_r2': eval_val['r2'],
            'final/val_rmse': eval_val['rmse'],
            'final/val_mde': eval_val['mde'],
            'final/test_mse': eval_test['mse'],
            'final/test_r2': eval_test['r2'],
            'final/test_rmse': eval_test['rmse'],
            'final/test_mde': eval_test['mde'],
        }
    )

    writer.flush()
    writer.close()
    print(f'TensorBoard logs for seed {seed} saved.')

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