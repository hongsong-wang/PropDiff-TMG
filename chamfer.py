import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from utils.utils import str2bool
import argparse
import sklearn
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from network.classifier_net import Classifier
from network.dual_encoder import VisionEncoder, TextEncoder
from generate import generate_based_on_text
from utils.mesh_utils import voxel2mesh

# Chamfer 距离计算
def chamfer_distance(pcd1, pcd2, batch_size=5000):
    
    device = pcd1.device
    N, M = pcd1.size(0), pcd2.size(0)
    # 计算 pcd1 中每个点到 pcd2 的最近距离
    min_dists1 = []
    for i in range(0, N, batch_size):
        batch = pcd1[i:i+batch_size]  # [B1, 3]
        dists = torch.cdist(batch, pcd2)  # [B1, M]
        min_dist = torch.min(dists, dim=1)[0]  # [B1]
        min_dists1.append(min_dist)
    dist1 = torch.cat(min_dists1)  # [N]
    # 计算 pcd2 中每个点到 pcd1 的最近距离
    min_dists2 = []
    for i in range(0, M, batch_size):
        batch = pcd2[i:i+batch_size]  # [B2, 3]
        dists = torch.cdist(batch, pcd1)  # [B2, N]
        min_dist = torch.min(dists, dim=1)[0]  # [B2]
        min_dists2.append(min_dist)
    dist2 = torch.cat(min_dists2)  # [M]
    # 返回 Chamfer 距离
    return torch.mean(dist1) + torch.mean(dist2)

def voxel_to_pointcloud(voxel, threshold=0.5):

    if voxel.dim() == 4:
        voxel = voxel[0]  # [D, H, W]

    coords = (voxel > threshold).nonzero(as_tuple=False).float()  # [N, 3]
    dims = voxel.shape  # 通常为 (64, 64, 64)
    coords = coords / torch.tensor(dims, dtype=torch.float32, device=voxel.device)  # 归一化到 [0, 1]
    # coords = coords*2.0 - 1.0
    return coords


# 修改计算 Chamfer 距离的函数
def calculate_chamfer_distance(real_voxels, fake_voxels, device):
    charm_score = 0
    for j in range(real_voxels.shape[0]):
        real_pointcloud = voxel_to_pointcloud(real_voxels[j], threshold=0.5).to(device)
        fake_pointcloud = voxel_to_pointcloud(fake_voxels[j], threshold=0.5).to(device)
        if real_pointcloud.shape[0] == 0 or fake_pointcloud.shape[0] == 0:
            continue        
        chamfer = chamfer_distance(real_pointcloud, fake_pointcloud, batch_size=5000).item()
        charm_score += chamfer
    return charm_score / real_voxels.shape[0]
# 解析命令行参数
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help='Dataset path')
    parser.add_argument('--model_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/results/debug/new8/best-loss-epoch=305-loss=0.1566.ckpt", help='model path')
    parser.add_argument("--output_path", type=str, default="text_results/")
    parser.add_argument('--classifier_ckpt', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier.pth", help='classifier path')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--tensor_w", type=float, default=2.5)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()
    return args


# 主程序
def main():
    args = parsing()
    device = args.device
    # 数据集和 DataLoader
    # 读取体素数据
    voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(np.float32)

    # 读取文本标签
    annotation_file = os.path.join(args.dataset_path, "captions.csv")
    annotations = pd.read_csv(annotation_file)["Captions"].tolist()  # 取出标题

    # 生成独热编码标签 (2000, 20)
    labels = np.zeros((2000, 20), dtype=np.float32)
    for i in range(20):
        labels[i * 100:(i + 1) * 100, i] = 1


    # 打乱索引
    index = np.arange(2000)
    np.random.seed(0)
    np.random.shuffle(index)

    # 处理体素数据
    voxels = torch.tensor(voxels, dtype=torch.float32).to(device)  # 添加通道维度

    chamfer = 0
    for idx, i in enumerate(index[:200]):
        generated_voxels, generated_voxels_bt = generate_based_on_text(model_path=args.model_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                                 num_generate=args.num_generate, truncated_time=args.truncated_time,
                                 query=annotations[i], w=args.tensor_w)
        chamfer += calculate_chamfer_distance(voxels[i].repeat(10,1,1,1), generated_voxels_bt.to(device), device)

        chamfer_score = chamfer / (idx + 1)
        print(f"epoch: {idx}, Chamfer Distance: {chamfer_score:.4f}")

        
    # 保存结果到txt
    output_metrics_path = os.path.join(args.output_path, "metrics_chamfer.txt")
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_metrics_path, "w") as f:
        f.write(f"Chamfer Distance: {chamfer_score:.4f}")
    print(f"Metrics saved to {output_metrics_path}")

if __name__ == '__main__':
    main()
