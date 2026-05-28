import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from network.data_loader_text import VoxelDataset

torch.set_num_threads(2)
# 3D 卷积分类器
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # unit 1, [64,64,64,1] => [32,32,32,16]
        self.conv1a = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv1b = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.max1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # unit 2, => [16,16,16,32]
        self.conv2a = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv2b = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.max2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # unit 3, => [8,8,8,64]
        self.conv3a = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.max3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # unit 4, => [4,4,4,128]
        self.conv4a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.max4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # unit 5, => [2,2,2,256]
        self.conv5a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.max5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # unit 6, => [1,1,1,512]
        self.conv6a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv6b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.max6 = nn.MaxPool3d(kernel_size=2, stride=2)

        # unit 7, 全连接层
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 20)  # 20 类分类

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.max1(x)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.max2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.max3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.max4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.max5(x)

        x = F.relu(self.conv6a(x))
        x = F.relu(self.conv6b(x))
        x = self.max6(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        label_predicted = self.fc3(x)

        return x, label_predicted  # 返回特征和预测结果



# 训练和验证代码
def train_and_validate(model, train_loader, val_loader, epochs=20, lr=0.0001, device="cuda"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for res in train_loader:
            voxels, labels = res['occupancy'].to(device), res['label'].to(device)

            optimizer.zero_grad()
            _, outputs = model(voxels)
            label = torch.argmax(labels, dim=1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(label).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = train_loss / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for res in val_loader:
                voxels, labels = res['occupancy'].to(device), res['label'].to(device)
                _, outputs = model(voxels)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier2.pth")


# 解析参数
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help='Dataset path')
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Training dataset split ratio")
    parser.add_argument('--epochs', type=int, default=100, help="Training epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()


def main():
    args = parsing()
    device = args.device
    # 数据集和 DataLoader
    dataset = VoxelDataset(args.dataset_path)
    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
   
    # 模型训练
    model = Classifier()
    train_and_validate(model, train_loader, val_loader, epochs=args.epochs, lr=0.0001)

if __name__ == "__main__":
    main()