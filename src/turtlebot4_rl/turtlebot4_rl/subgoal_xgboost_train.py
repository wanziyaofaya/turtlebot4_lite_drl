import os
import random
from datetime import datetime
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import xgboost as xgb
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
    print(f'Starting XGBoost Training for SEED: {seed}')
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
    writer_path = f'runs/xgboost_multi_seed_{run_timestamp}/seed_{seed}'
    writer = SummaryWriter(log_dir=writer_path)

    # XGBoost 参数
    xgb_params = {
        'n_estimators': 180,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': seed,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
    }

    # 准备验证集
    Y_val_normalized = (data_numpy['val']['y'] - regression_label_stats.mean) / regression_label_stats.std

    def evaluate(part: str, models_list: list) -> dict:
        """评估模型性能"""
        y_pred_normalized = np.column_stack(
            [m.predict(data_numpy[part]['x_num']) for m in models_list]
        )

        n_pred_outputs = y_pred_normalized.shape[1]
        y_pred = (
            y_pred_normalized * regression_label_stats.std[:n_pred_outputs]
            + regression_label_stats.mean[:n_pred_outputs]
        )
        y_true = data_numpy[part]['y'][:, :n_pred_outputs]

        mde = (
            float(np.mean(np.linalg.norm(y_pred - y_true, axis=1)))
            if n_pred_outputs == 2
            else float('nan')
        )
        
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        rmse = mse ** 0.5
        return {'mse': mse, 'r2': r2, 'rmse': rmse, 'mde': mde}

    class TensorBoardCallback(xgb.callback.TrainingCallback):
        def __init__(self, writer, target_idx, data_numpy, regression_label_stats, models_so_far, n_outputs):
            self.writer = writer
            self.target_idx = target_idx
            self.data_numpy = data_numpy
            self.regression_label_stats = regression_label_stats
            self.models_so_far = models_so_far
            self.n_outputs = n_outputs

        def after_iteration(self, model, epoch, evals_log):
            global_step = epoch
            booster = model
            mean_i = self.regression_label_stats.mean[self.target_idx]
            std_i = self.regression_label_stats.std[self.target_idx]

            for part in ['train', 'val', 'test']:
                x = self.data_numpy[part]['x_num']
                y_true_raw = self.data_numpy[part]['y'][:, self.target_idx]

                dmat = xgb.DMatrix(x)
                y_pred_norm = booster.predict(dmat)
                y_pred = y_pred_norm * std_i + mean_i

                mse = sklearn.metrics.mean_squared_error(y_true_raw, y_pred)
                r2 = sklearn.metrics.r2_score(y_true_raw, y_pred)
                rmse = mse ** 0.5

                self.writer.add_scalar(f'target_{self.target_idx}/{part}_mse', mse, global_step)
                self.writer.add_scalar(f'target_{self.target_idx}/{part}_rmse', rmse, global_step)
                self.writer.add_scalar(f'target_{self.target_idx}/{part}_r2', r2, global_step)

            num_trained_targets = len(self.models_so_far) + 1
            if num_trained_targets == self.n_outputs:
                boosters = [m.get_booster() for m in self.models_so_far] + [booster]

                for part in ['train', 'val', 'test']:
                    x = self.data_numpy[part]['x_num']
                    y_true_all = self.data_numpy[part]['y'][:, :num_trained_targets]

                    dmat = xgb.DMatrix(x)
                    preds = []
                    for j, b in enumerate(boosters):
                        mean_j = self.regression_label_stats.mean[j]
                        std_j = self.regression_label_stats.std[j]
                        y_pred_norm_j = b.predict(dmat)
                        y_pred_j = y_pred_norm_j * std_j + mean_j
                        preds.append(y_pred_j)
                    y_pred_all = np.column_stack(preds)

                    mse = sklearn.metrics.mean_squared_error(y_true_all, y_pred_all)
                    r2 = sklearn.metrics.r2_score(y_true_all, y_pred_all)
                    rmse = mse ** 0.5
                    mde = float(np.mean(np.linalg.norm(y_pred_all - y_true_all, axis=1)))

                    self.writer.add_scalar(f'{part}/mse', mse, global_step)
                    self.writer.add_scalar(f'{part}/rmse', rmse, global_step)
                    self.writer.add_scalar(f'{part}/mde', mde, global_step)
                    self.writer.add_scalar(f'{part}/r2', r2, global_step)
            
            self.writer.flush()
            return False

    models = []
    for i in range(n_outputs):
        print(f"\nTraining model for target {i+1}/{n_outputs}...")
        
        callback = TensorBoardCallback(writer, i, data_numpy, regression_label_stats, models, n_outputs)
        
        model_i = xgb.XGBRegressor(
            **xgb_params,
            callbacks=[callback],
        )
        model_i.fit(
            data_numpy['train']['x_num'],
            Y_train_normalized[:, i],
            eval_set=[(data_numpy['val']['x_num'], Y_val_normalized[:, i])],
            verbose=True,
        )
        models.append(model_i)
        
        eval_train_i = evaluate('train', models)
        eval_val_i = evaluate('val', models)
        eval_test_i = evaluate('test', models)
        
        writer.add_scalar(f'overall/train_mse_after_target_{i}', eval_train_i['mse'], i)
        writer.add_scalar(f'overall/val_mse_after_target_{i}', eval_val_i['mse'], i)
        writer.add_scalar(f'overall/test_mse_after_target_{i}', eval_test_i['mse'], i)

        if not np.isnan(eval_train_i['mde']):
            writer.add_scalar(f'overall/train_mde_after_target_{i}', eval_train_i['mde'], i)
            writer.add_scalar(f'overall/val_mde_after_target_{i}', eval_val_i['mde'], i)
            writer.add_scalar(f'overall/test_mde_after_target_{i}', eval_test_i['mde'], i)

    eval_train = evaluate('train', models)
    eval_val = evaluate('val', models)
    eval_test = evaluate('test', models)

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

    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'subgoal_xgboost_{run_timestamp}_seed{seed}.joblib')

    model_paths = []
    for i, m in enumerate(models):
        model_file = os.path.join(model_save_dir, f'subgoal_xgboost_{run_timestamp}_seed{seed}_target_{i}.json')
        booster = m.get_booster()
        booster.save_model(model_file)
        model_paths.append(model_file)

    joblib.dump(
        {
            'model_paths': model_paths,
            'preprocessing': preprocessing,
            'regression_label_stats': regression_label_stats,
            'task_type': task_type,
            'timestamp': run_timestamp,
            'seed': seed,
            'metrics': {'train': eval_train, 'val': eval_val, 'test': eval_test},
            'xgb_params': xgb_params,
        },
        model_save_path,
    )
    print(f'\nModel saved to {model_save_path}')

    writer.add_hparams(
        {'model': 'XGBoost', 'n_estimators': xgb_params['n_estimators'], 
         'max_depth': xgb_params['max_depth'], 'learning_rate': xgb_params['learning_rate'], 'seed': seed},
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
        },
    )
    writer.flush()
    writer.close()
    print(f'TensorBoard logs for target seed {seed} saved.')

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
