import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

class SubgoalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            header = f.readline()
            for line in f:
                items = line.strip().split(',')
                # 一行包含: 起点2 + 终点2 + 子目标2 + 激光64 = 70 列
                if len(items) < 70:
                    continue
                # 起点(2), 终点(2), 子目标点(2), 激光(64)
                start = [float(items[0]), float(items[1])]
                goal = [float(items[2]), float(items[3])]
                subgoal = [float(items[4]), float(items[5])]
                lidar = [float(x) for x in items[6:70]]
                x = start + goal + lidar  # 输入: 2+2+64=68
                y = subgoal              # 输出: 2
                self.data.append((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class SubgoalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(68, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_subgoal_net(dataset_path, epochs=800, batch_size=128, lr=1e-3,
                      model_save_path='subgoal_net.pth', train_ratio=0.8, patience=30,
                      log_dir='tensorboard_logs/subgoal_net'):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SubgoalDataset(dataset_path)
    if len(dataset) < 2:
        raise ValueError("数据集太小，无法拆分训练/验证集")

    # 拆分训练/验证集
    train_len = int(len(dataset) * train_ratio)
    # 保证至少有一个样本在验证集中
    if train_len >= len(dataset):
        train_len = len(dataset) - 1
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SubgoalNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    wait = 0

    # TensorBoard writer：确保日志目录存在并创建 SummaryWriter
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    # 确保保存路径目录存在
    save_dir = os.path.dirname(model_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        # 验证
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = xv.to(device)
                yv = yv.to(device)
                pv = model(xv)
                val_loss_sum += criterion(pv, yv).item() * xv.size(0)
        val_loss = val_loss_sum / len(val_ds)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 写入 TensorBoard（每个 epoch）
        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch+1)
            writer.add_scalar('loss/val', val_loss, epoch+1)
            try:
                current_lr = optimizer.param_groups[0].get('lr', None)
                if current_lr is not None:
                    writer.add_scalar('lr', current_lr, epoch+1)
            except Exception:
                pass

        # 早停与保存最佳模型
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_save_path)
            wait = 0
            print(f"  Best model saved (val_loss={best_val:.6f}) -> {model_save_path}")
            if writer is not None:
                writer.add_scalar('loss/best_val', best_val, epoch+1)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

    # 关闭 TensorBoard writer
    if writer is not None:
        writer.close()

    print("Training finished.")

if __name__ == '__main__':
    train_subgoal_net('models/improved_astar_subgoal_dataset.txt', epochs=800, batch_size=128, lr=1e-3, model_save_path='models/improved_astar_subgoal_net.pth')

# python3 src/turtlebot4_rl/turtlebot4_rl/subgoal_net_train.py