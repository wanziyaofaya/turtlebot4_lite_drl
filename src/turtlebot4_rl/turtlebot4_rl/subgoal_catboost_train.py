import os
import random
from datetime import datetime
from typing import Literal, NamedTuple

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

file_path = 'models/subgoal_dataset.txt'
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

# 数据划分
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

# 特征预处理
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

print("\nStarting CatBoost Training...")
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# TensorBoard writer
writer = SummaryWriter(log_dir=f'runs/catboost_{run_timestamp}')

# CatBoost 参数
catboost_params = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'loss_function': 'RMSE',
    'early_stopping_rounds': 50,
    'verbose': 100,
    'task_type': 'GPU' if os.environ.get('USE_GPU', '0') == '1' else 'CPU',
}

# 准备验证集
Y_val_normalized = (data_numpy['val']['y'] - regression_label_stats.mean) / regression_label_stats.std

def evaluate(part: str, models_list: list) -> dict:
    """评估模型性能"""
    y_pred_normalized = np.column_stack([m.predict(data_numpy[part]['x_num']) for m in models_list])
    y_pred = y_pred_normalized * regression_label_stats.std + regression_label_stats.mean
    y_true = data_numpy[part]['y']
    
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    rmse = mse ** 0.5
    return {'mse': mse, 'r2': r2, 'rmse': rmse}

# 为每个输出目标训练独立的模型
models = []
for i in range(n_outputs):
    print(f"\nTraining model for target {i+1}/{n_outputs}...")
    model_i = CatBoostRegressor(**catboost_params)
    
    train_pool = Pool(data_numpy['train']['x_num'], Y_train_normalized[:, i])
    val_pool = Pool(data_numpy['val']['x_num'], Y_val_normalized[:, i])
    
    model_i.fit(train_pool, eval_set=val_pool)
    models.append(model_i)
    
    # 记录每个目标模型的训练历史到 TensorBoard
    evals_result = model_i.get_evals_result()
    if 'validation' in evals_result and 'RMSE' in evals_result['validation']:
        for epoch, rmse_val in enumerate(evals_result['validation']['RMSE']):
            writer.add_scalar(f'target_{i}/val_rmse', rmse_val, epoch)
    
    # 每个目标模型训练完后记录当前整体指标
    eval_train_i = evaluate('train', models)
    eval_val_i = evaluate('val', models)
    eval_test_i = evaluate('test', models)
    
    writer.add_scalar(f'overall/train_mse_after_target_{i}', eval_train_i['mse'], i)
    writer.add_scalar(f'overall/val_mse_after_target_{i}', eval_val_i['mse'], i)
    writer.add_scalar(f'overall/test_mse_after_target_{i}', eval_test_i['mse'], i)

# 最终评估
eval_train = evaluate('train', models)
eval_val = evaluate('val', models)
eval_test = evaluate('test', models)

print('\n' + '='*40)
print('CatBoost RESULTS')
print('='*40)
print(f'Train - MSE: {eval_train["mse"]:.6f} | R²: {eval_train["r2"]:.6f} | RMSE: {eval_train["rmse"]:.6f}')
print(f'Val   - MSE: {eval_val["mse"]:.6f} | R²: {eval_val["r2"]:.6f} | RMSE: {eval_val["rmse"]:.6f}')
print(f'Test  - MSE: {eval_test["mse"]:.6f} | R²: {eval_test["r2"]:.6f} | RMSE: {eval_test["rmse"]:.6f}')

# 保存模型
model_save_dir = 'models'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, f'subgoal_catboost_{run_timestamp}.joblib')

joblib.dump(
    {
        'models': models,
        'preprocessing': preprocessing,
        'regression_label_stats': regression_label_stats,
        'task_type': task_type,
        'timestamp': run_timestamp,
        'metrics': {'train': eval_train, 'val': eval_val, 'test': eval_test},
    },
    model_save_path,
)
print(f'\nModel saved to {model_save_path}')

# 记录最终指标到 TensorBoard
writer.add_hparams(
    {'model': 'CatBoost', 'iterations': catboost_params['iterations'], 
     'depth': catboost_params['depth'], 'learning_rate': catboost_params['learning_rate']},
    {'final/train_mse': eval_train['mse'], 'final/train_r2': eval_train['r2'], 'final/train_rmse': eval_train['rmse'],
     'final/val_mse': eval_val['mse'], 'final/val_r2': eval_val['r2'], 'final/val_rmse': eval_val['rmse'],
     'final/test_mse': eval_test['mse'], 'final/test_r2': eval_test['r2'], 'final/test_rmse': eval_test['rmse']}
)

writer.close()
print('TensorBoard logs saved.')
