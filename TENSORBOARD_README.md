# Tensorboard集成说明

## 概述

本项目已经集成了Tensorboard支持，用于可视化强化学习训练过程中的指标。训练过程中会自动记录以下信息：

- **episode/reward**: 每个episode的总奖励
- **episode/length**: 每个episode的长度
- **episode/reward_avg_10**: 最近10个episode的平均奖励
- **episode/length_avg_10**: 最近10个episode的平均长度
- **其他标准RL指标**: 由stable-baselines3自动记录的标准指标

## 使用方法

### 1. 运行训练
```bash
python src/turtlebot4_rl/turtlebot4_rl/rl_node.py --algorithm PPO --timesteps 10000 --episodes 5
```

### 2. 启动Tensorboard

有两种方式启动Tensorboard：

**方式1: 使用提供的脚本**
```bash
./start_tensorboard.sh
```

**方式2: 手动启动**
```bash
~/.local/bin/tensorboard --logdir tensorboard_logs
```

### 3. 查看训练结果

打开浏览器访问 `http://localhost:6006`

## 日志结构

Tensorboard日志按以下结构保存：
```
tensorboard_logs/
├── PPO/
│   └── YYYYMMDD_HHMMSS/
│       ├── task_1/
│       ├── task_2/
│       └── ...
```

## 可视化指标

在Tensorboard中可以看到：

1. **SCALARS**: 
   - 训练loss
   - episode奖励和长度
   - 学习率变化
   - 其他训练指标

2. **GRAPHS**: 模型架构图（如果支持）

3. **DISTRIBUTIONS/HISTOGRAMS**: 网络权重分布（如果启用）

## 故障排除

### Tensorboard未找到
如果出现"tensorboard命令未找到"的错误：
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 端口被占用
如果6006端口被占用，可以指定其他端口：
```bash
~/.local/bin/tensorboard --logdir tensorboard_logs --port 6007
```

### 清理旧日志
如果想清理旧的训练日志：
```bash
rm -rf tensorboard_logs/
```

## 高级用法

### 比较不同算法
可以在Tensorboard中同时加载多个算法的日志：
```bash
~/.local/bin/tensorboard --logdir tensorboard_logs
```
然后在界面中使用左侧的勾选框来选择要显示的训练会话。

### 自定义指标
如果需要记录额外的自定义指标，可以在`TensorboardCallback`类中添加：
```python
self.logger.record('custom/metric_name', metric_value)
```