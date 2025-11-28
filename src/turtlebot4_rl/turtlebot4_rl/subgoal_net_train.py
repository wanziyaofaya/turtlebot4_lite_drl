import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor  # 关键导入
from tabpfn import TabPFNRegressor

import huggingface_hub

# 将下方的 "hf_xxxxxxxx" 替换为你第二步里复制的真实 Token
# 注意：不要把带有真实 Token 的代码发给别人看
huggingface_hub.login(token="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")

# 1. 读取数据
df = pd.read_csv('improved_astar_subgoal_dataset.txt')

# 2. 准备 X 和 y (双目标)
# 定义你的两个目标列名
target_cols = ['subgoal_x', 'subgoal_y']

# X: 删除这两个目标列
X = df.drop(columns=target_cols)

# y: 提取这两个列，y 现在的形状是 (n_samples, 2)
y = df[target_cols]

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 4. 初始化模型 (关键步骤)
# TabPFN 一次只能预测一列，所以我们需要用 MultiOutputRegressor 把它包起来
# 这样它内部会自动建立两个 TabPFN 模型，分别预测 x 和 y
base_regressor = TabPFNRegressor(device="cuda") 
regressor = MultiOutputRegressor(base_regressor)

regressor.fit(X_train, y_train)

# 5. 预测
predictions = regressor.predict(X_test)

# predictions 也是一个 (n_samples, 2) 的数组
# 第一列是 subgoal_x 的预测值，第二列是 subgoal_y 的预测值

# 6. 评估
# calculate metrics generally (average across outputs)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Overall Mean Squared Error:", mse)
print("Overall R² Score:", r2)

# ---------------------------------------------------------
# 如果你想分别查看 subgoal_x 和 subgoal_y 的准确度：
# ---------------------------------------------------------
print("-" * 30)
print("Detailed Metrics:")
# multioutput='raw_values' 会返回每一列的独立得分
r2_individual = r2_score(y_test, predictions, multioutput='raw_values')
mse_individual = mean_squared_error(y_test, predictions, multioutput='raw_values')

print(f"R² for {target_cols[0]}: {r2_individual[0]:.4f}")
print(f"R² for {target_cols[1]}: {r2_individual[1]:.4f}")
print(f"MSE for {target_cols[0]}: {mse_individual[0]:.4f}")
print(f"MSE for {target_cols[1]}: {mse_individual[1]:.4f}")