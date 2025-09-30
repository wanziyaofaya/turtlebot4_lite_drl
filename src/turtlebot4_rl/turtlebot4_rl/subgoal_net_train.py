import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SubgoalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            header = f.readline()
            for line in f:
                items = line.strip().split(',')
                if len(items) < 646:
                    continue
                # 起点(2), 终点(2), 子目标点(2), 激光(640)
                start = [float(items[0]), float(items[1])]
                goal = [float(items[2]), float(items[3])]
                subgoal = [float(items[4]), float(items[5])]
                lidar = [float(x) for x in items[6:646]]
                x = start + goal + lidar  # 输入: 2+2+640=644
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
            nn.Linear(644, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_subgoal_net(dataset_path, epochs=50, batch_size=32, lr=1e-3, model_save_path='subgoal_net.pth'):
    dataset = SubgoalDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SubgoalNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_subgoal_net('models/PPO/subgoal_dataset.txt', epochs=50, batch_size=32, lr=1e-3, model_save_path='models/PPO/subgoal_net.pth')

# python3 src/turtlebot4_rl/turtlebot4_rl/subgoal_net_train.py