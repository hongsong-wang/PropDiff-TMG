import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import datetime 
from network.data_loader_text import VoxelDataset, VoxelDataset1

torch.set_num_threads(2)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 3D 卷积分类器
class Slover(nn.Module):
    def __init__(self):
        super(Slover, self).__init__()

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
        self.fc3 = nn.Linear(1024, 3)  # 3 类分类

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

        return label_predicted  # 返回特征和预测结果

# 解析参数
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help='Dataset path')
    parser.add_argument('--voxel_folder', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress", help='Dataset path')
    parser.add_argument('--csv_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_npr.csv", help='Dataset path')
    parser.add_argument('--output_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/training_result", help='Output path')
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Training dataset split ratio")
    parser.add_argument('--epochs', type=int, default=200, help="Training epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

# 训练主函数
def main():
    args = parsing()
    device = args.device
    # 数据集和 DataLoader
    # dataset = VoxelDataset(args.dataset_path,
    #                     transform=None,
    #                     use_tensor_condition=True)
    dataset = VoxelDataset1(csv_path=args.csv_path,
                                voxel_folder=args.voxel_folder,
                                use_tensor_condition=True
                                )
    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
   
    # 设置输出路径
    output_path = os.path.join(args.output_path, "solver")
    create_dir(output_path)

    # phi = dataset.prop[:, 0]
    # E_label = dataset.prop[:, 1]
    # Ani_label = dataset.prop[:, 2]

    # # print(phi.shape)
    # # print(E_label.shape)

    # # 可视化数据分布
    # fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    # axs[0].hist(E_label)
    # axs[1].hist(Ani_label)
    # axs[2].hist(phi)
    # plt.savefig(os.path.join(output_path, "distribution.png"))
    # plt.close(fig)

    # 8. 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Slover().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss(reduction='sum')

    # 9. 训练循环
    start_time = time.time()
    acc_file = open(os.path.join(output_path, "accuracy.txt"), 'w')
    acc_file.write("epoch loss acc\n")
    best_val = 100
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            x, y = data["occupancy"].to(device), data["prop"].to(device)
            # print(x.shape)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch}: train loss: {train_loss:.6f}")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in train_loader:
                x, y = data["occupancy"].to(device), data["prop"].to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}: val loss: {val_loss:.6f}")

        acc_file.write(f"{epoch} {train_loss:.6f} {val_loss:.6f}\n")
        # 保存模型
        os.makedirs(os.path.join(output_path, 'ckpt'), exist_ok=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_path, f'ckpt/solver_newdata_best.pth'))
        if epoch == args.epochs-1:
            torch.save(model.state_dict(), os.path.join(output_path, f'ckpt/solver_newdata_last.pth'))

    # 记录训练信息
    acc_file.write(f"best_epoch: {best_epoch}\n")
    acc_file.write(str(datetime.datetime.now()) + "\n")
    acc_file.write(f"PyTorch version: {torch.__version__}\n")
    acc_file.write(f"Training time: {time.time() - start_time:.2f} seconds\n")
    acc_file.close()

if __name__ == '__main__':
    main()
    
    
    
    
    