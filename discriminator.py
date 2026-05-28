import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from network.model_trainer import DiffusionModel
from FID import VoxelDatasetClean
from tqdm import tqdm

class PatchVoxelDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm3d(256)

        self.patch_classifier = nn.Conv3d(256, 1, kernel_size=1)  # patch logits

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_global = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        feat = F.relu(self.bn4(self.conv4(x)))

        patch_logits = self.patch_classifier(feat)  # (B,1,d,d,d)
        pooled = self.global_pool(feat).view(batch_size, -1)
        global_logit = self.fc_global(pooled)       # (B,1)

        patch_prob = torch.sigmoid(patch_logits).mean(dim=[1,2,3,4])  # (B,)
        global_prob = torch.sigmoid(global_logit).squeeze(1)          # (B,)
        final_score = (patch_prob + global_prob) / 2                  # (B,)

        return patch_logits, global_logit, final_score

class VoxelDiscriminator(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        # 使用 Classifier 网络的卷积部分（特征提取部分）
        self.conv1a = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1)
        self.conv1b = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.max1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv2b = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.max2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.max3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.max4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.max5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv6a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv6b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.max6 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 特征展平并传入判别网络部分
        self.fc1 = nn.Linear(512 * 1 * 1 * 1, 1024)
        self.dropout = nn.Dropout(0.5)
        
        # 判别器的输出部分，用于输出真假（单一值）
        self.fc2 = nn.Linear(1024, 1)  # 输出真假值，0 或 1

    def forward(self, x):
        # 卷积层部分：特征提取
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

        # 展平特征并通过全连接层
        x = x.view(x.size(0), -1)  # 展平为 1D 向量
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # 输出判别真假值

        return x  # 返回一个值，用于真假判定（0 或 1）


class DiscriminatorTrainer:
    def __init__(self, lr=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = PatchVoxelDiscriminator().to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, real_voxels, fake_voxels):
        real_voxels = real_voxels.to(self.device)
        fake_voxels = fake_voxels.to(self.device)

        B = real_voxels.size(0)
        real_labels = torch.ones((B, 1), device=self.device)
        fake_labels = torch.zeros((B, 1), device=self.device)

        # Forward pass
        real_patch, real_global, real_score = self.discriminator(real_voxels)
        fake_patch, fake_global, fake_score = self.discriminator(fake_voxels)

        # ---- Patch losses ----
        loss_real_patch = self.criterion(real_patch, torch.ones_like(real_patch))
        loss_fake_patch = self.criterion(fake_patch, torch.zeros_like(fake_patch))

        # ---- Global losses ----
        loss_real_global = self.criterion(real_global, real_labels)
        loss_fake_global = self.criterion(fake_global, fake_labels)

        # ---- Total loss ----
        loss = (
            loss_real_patch + loss_fake_patch +
            loss_real_global + loss_fake_global
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "d_loss": loss.item(),
            "real_score": real_score.mean().item(),
            "fake_score": fake_score.mean().item()
        }



@torch.no_grad()
def discriminator_score(discriminator, generated_voxels):
    discriminator.eval()
    scores = discriminator(generated_voxels).cpu().numpy()
    return {
        "D_mean": float(scores.mean()),
        "D_std": float(scores.std()),
        "D_score": round(1.0 - scores.mean(), 4)  # 越接近 0 越差，越接近 1 越真实
    }

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help='Dataset path')
    parser.add_argument('--model_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/results/debug/old/best-loss-epoch=451-loss=0.1319_1.ckpt", help='model path')
    parser.add_argument("--output_path", type=str, default="text_results/")
    parser.add_argument('--classifier_ckpt', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier.pth", help='classifier path')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument("--num_generate", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--tensor_w", type=float, default=1.5)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()
    return args

def main():
    args = parsing()
    device = args.device

    # 数据集和 DataLoader
    # 读取体素数据
    # voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(np.float32)
    # voxels = voxels*2-1
    # # 读取文本标签
    # annotation_file = os.path.join(args.dataset_path, "captions.csv")
    # annotations = pd.read_csv(annotation_file)["Captions"].tolist()  # 取出标题
    dataset = VoxelDatasetClean("/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_npr.csv", "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress")
    total_len = len(dataset)
    num_samples = 2000
    np.random.seed(0)
    indices = np.random.permutation(total_len)[:num_samples]

    voxels = []
    annotations = []
    props = []

    for idx in indices:
        print(idx)
        res = dataset[idx]
        prop_str = f"phi:{res['prop'][0]:.4f} E:{res['prop'][1]:.4f} v:{res['prop'][2]:.4f}"
        # 拼接到原始文本
        caption = res['caption'] + " " + prop_str   
        voxels.append(res['occupancy'])
        annotations.append(caption)
        props.append(res['prop'])

    voxels = np.array(voxels).astype(np.float32)
    voxels = (voxels+1) / 2
    prop = np.array(props)

    fake_voxels = np.load("/home/daibingxuan/workspace/microstructure_generation_3d/fake_voxels_res.npy")
    fake_voxels = torch.tensor(fake_voxels).to(device)
    
    # 加载生成器模型
    discrete_diffusion = DiffusionModel.load_from_checkpoint(args.model_path).to('cuda:4')
    ema = True  # 是否使用 EMA 模型
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model

    # 初始化判别器和训练器
    discriminator = VoxelDiscriminator().to(device)
    trainer = DiscriminatorTrainer(
        lr=1e-4
    )

    for epoch in range(50):
        # 每次训练时生成一个批次的体素
        for i in range(0, 2000, args.batch_size):
            # 获取当前批次的文本数据
            # batch_annotations = annotations[i:i+args.batch_size]
            # 生成虚假体素
            # fake_voxels = generator.sample_with_text(batch_annotations, prop=None, batch_size=len(batch_annotations),
                                                    # steps=args.steps, truncated_index=args.truncated_time, tensor_w=args.tensor_w).to(device)
            fake_voxel = fake_voxels[i:i+args.batch_size]
            noise_strength = min(0.5, 1.0 / (epoch + 1))
            if epoch >= 15:
                noise_strength=0
            fake_voxel += torch.randn_like(fake_voxel) * noise_strength
            fake_voxel[fake_voxel > 0] = 1
            fake_voxel[fake_voxel < 0] = 0
            # fake_voxels = torch.sigmoid(fake_voxels)
            # 获取真实体素
            real_voxels = torch.tensor(voxels[i:i+args.batch_size]).to(device)
            # print(len(real_voxels))
            # 训练判别器
            result = trainer.train_step(real_voxels, fake_voxel)

            # 打印训练过程
            print(f"Epoch [{epoch+1}/{50}], Step [{i//args.batch_size + 1}/{2000//args.batch_size}], "
                  f"D Loss: {result['d_loss']:.4f}, Real: {result['real_score']:.4f}, Fake: {result['fake_score']:.4f}")

        # 保存已训练的判别器模型
        torch.save(trainer.discriminator.state_dict(), f"discriminator_epoch_new.pth")

    # 最后使用判别器打分一个批次

if __name__ == "__main__":
    main()
