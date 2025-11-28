#!/bin/bash

# 启动Tensorboard脚本
# 使用方法: ./start_tensorboard.sh
# tensorboard --logdir tensorboard_logs/subgoal_net_v2
echo "启动Tensorboard..."
echo "日志目录: $(pwd)/tensorboard_logs"

# 检查是否安装了tensorboard
if ! command -v ~/.local/bin/tensorboard &> /dev/null
then
    echo "Tensorboard 未安装，正在安装..."
    pip install tensorboard --break-system-packages
fi

# 创建tensorboard_logs目录（如果不存在）
mkdir -p tensorboard_logs

# 启动Tensorboard
echo "Tensorboard正在启动..."
echo "打开浏览器访问: http://localhost:6006"
~/.local/bin/tensorboard --logdir tensorboard_logs --host 0.0.0.0