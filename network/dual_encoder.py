import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.video import R3D_18_Weights
import pandas as pd
import numpy as np
import time
import random
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import timm
from network.data_loader_text import VoxelDataset, VoxelDataset1
import torchvision.models.video as models

torch.set_num_threads(2)


class ProjectionLayer(nn.Module):
    def __init__(self, num_projection_layers, input_dims, projection_dims, dropout_rate):
        super(ProjectionLayer, self).__init__()

        self.input_proj = nn.Linear(input_dims, projection_dims)
        self.activation = nn.GELU()
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(projection_dims, projection_dims),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ) for _ in range(num_projection_layers)
        ])
        self.normalization = nn.LayerNorm(projection_dims)  # 共享 LayerNorm

    def forward(self, x):
        x = self.activation(self.input_proj(x)) 
        for layer in self.projection_layers:
            x = x + layer(x) 
            x = self.normalization(x) 
        return x


class VisionEncoder(nn.Module):
    def __init__(self, num_projection_layers, input_dims, projection_dims, dropout_rate, trainable=False):
        super(VisionEncoder, self).__init__()
        print("start load visionencoder model>>>")
        self.resnet3d = models.r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.resnet3d.stem[0] = nn.Conv3d(
            in_channels=1,  # 改为单通道
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        )
        self.resnet3d.fc = nn.Identity()

        for param in self.resnet3d.parameters():
            param.requires_grad = False

        # 添加投影层
        self.projection = ProjectionLayer(num_projection_layers, input_dims, projection_dims, dropout_rate)
        print("VisionEncoder with r3D model loaded successfully!")

    def forward(self, x):
        x = self.resnet3d(x)  # 提取 3D 特征
        x = x.view(x.shape[0], -1)  # 展平
        x1 = x
        x = self.projection(x)  # 通过投影层
        return x1, x


class TextEncoder(nn.Module):
    def __init__(self, num_projection_layers, input_dims, projection_dims, dropout_rate, trainable=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/home/daibingxuan/workspace/microstructure_generation_3d/bert_localpath")
        self.bert = AutoModelForMaskedLM.from_pretrained("/home/daibingxuan/workspace/microstructure_generation_3d/bert_localpath")
        
        for param in self.bert.parameters():
            param.requires_grad = trainable
        
        # 添加 ProjectionLayer 处理特征
        self.projection = ProjectionLayer(num_projection_layers, input_dims, projection_dims, dropout_rate)
        print("TextEncoder load successful!")
    def forward(self, text, prop):
        
        encoded = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.bert.device)
        attention_mask = encoded["attention_mask"].to(self.bert.device)
        
        # BERT 模型输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        # # 提取 [CLS] 向量
        cls_output = last_hidden_state[:, 0, :]
        # 通过投影层进一步处理
        x = self.projection(cls_output)
        
        return x


class DualEncoder(nn.Module):
    def __init__(self, text_encoder, vision_encoder, temperature=1.0):
        super().__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, captions, prop, images):
        caption_embeddings = self.text_encoder(captions, prop)
        _, image_embeddings = self.vision_encoder(images)
        return caption_embeddings, image_embeddings

    # def compute_loss(self, caption_embeddings, image_embeddings):
    #     # normalize
    #     caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)
    #     image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

    #     # logits
    #     logits = torch.matmul(caption_embeddings, image_embeddings.T) / self.temperature

    #     # labels 0..batch_size-1
    #     labels = torch.arange(len(logits)).long().to(logits.device)

    #     # symmetric CLIP loss
    #     loss_i2t = F.cross_entropy(logits, labels)
    #     loss_t2i = F.cross_entropy(logits.T, labels)
    #     loss = (loss_i2t + loss_t2i) / 2
    #     return loss
    def compute_loss(self, caption_embeddings, image_embeddings):
    
        caption_embeddings = F.normalize(caption_embeddings, p=2, dim=1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)    
        # 计算logits（点积除以温度）
        logits = torch.matmul(caption_embeddings, image_embeddings.T) / self.temperature
        # 计算图像与图像的相似度矩阵
        images_similarity = torch.matmul(image_embeddings, image_embeddings.T)
        # 计算文本与文本的相似度矩阵
        captions_similarity = torch.matmul(caption_embeddings, caption_embeddings.T)
        # 目标分布：(captions_similarity + images_similarity) / (2 * temperature) 的 softmax
        targets = torch.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature),
            dim=-1
        )
        # 计算文本方向的交叉熵损失
        captions_log_probs = F.log_softmax(logits, dim=1)
        captions_loss = -torch.sum(targets * captions_log_probs, dim=1).mean()
        
        # 计算图像方向的交叉熵损失
        images_log_probs = F.log_softmax(logits.T, dim=1)
        images_loss = -torch.sum(targets.T * images_log_probs, dim=1).mean()
        
        return (captions_loss + images_loss) / 2

    def training_step(self, captions, prop, images, optimizer):
        optimizer.zero_grad()
        caption_embeddings, image_embeddings = self(captions, prop, images)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        loss.backward()
        optimizer.step()
        return loss.item()

class EarlyStopping:
    """ 早停，当验证损失不再改善时停止训练 """
    def __init__(self, patience=5, delta=0, restore_best_weights=True):
        """
        参数：
        - patience: 多少个 epoch 内验证损失未改善则停止训练
        - delta: 最小改善量
        - restore_best_weights: 是否恢复最佳模型权重
        """
        self.patience = patience
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """ 在每个 epoch 结束时调用 """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs.")
                if self.restore_best_weights and self.best_model_state:
                    model.load_state_dict(self.best_model_state)
                return True  # 停止训练
        return False  # 继续训练


# 训练代码
def train_model(model, train_loader, valid_loader, scheduler, optimizer, num_epochs=200, device="cuda"):
    history = {'train_loss': [], 'valid_loss': []}
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for res in tqdm(train_loader):
            voxel = res['occupancy'].to(device)
            captions = res['caption']
            prop = None
            loss = model.training_step(captions, prop, voxel, optimizer)
            running_train_loss += loss

        avg_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 验证
        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            for res in valid_loader:
                voxel = res['occupancy'].to(device)
                captions = res['caption']
                prop = None
                caption_embeddings, image_embeddings = model(captions, prop, voxel)
                cosine_sim = F.cosine_similarity(image_embeddings, caption_embeddings)
                print(cosine_sim)
                loss = model.compute_loss(caption_embeddings, image_embeddings)
                running_valid_loss += loss.item()

        avg_valid_loss = running_valid_loss / len(valid_loader)
        scheduler.step(avg_valid_loss)
        history['valid_loss'].append(avg_valid_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        if early_stopping(avg_valid_loss, model):
            break

    return history


def compute_top_k_accuracy(dataloader, batch_size, vision_encoder,
                                            text_encoder, k=500, device='cuda', normalize=True):
    all_voxel_embeddings = []
    all_text_embeddings = []
    
    vision_encoder.eval()  # Set to eval mode
    text_encoder.eval()    # Set to eval mode
    
    with torch.no_grad():
        for res in dataloader:
            voxel = res['occupancy'].to(device)
            caption = res['caption']
            prop = res['prop']
            # Compute embeddings
            _, voxel_embeddings = vision_encoder(voxel)
            text_embeddings = text_encoder(caption, prop)
            
            # Save embeddings
            all_voxel_embeddings.append(voxel_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
    
    # Concatenate embeddings
    all_voxel_embeddings = torch.cat(all_voxel_embeddings, dim=0).to(device)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(device)
    print(all_text_embeddings.min(), all_text_embeddings.max())
    print(all_voxel_embeddings.min(), all_voxel_embeddings.max())
    # Normalize embeddings
    if normalize:
        all_text_embeddings = torch.nn.functional.normalize(all_text_embeddings, p=2, dim=1)
        all_voxel_embeddings = torch.nn.functional.normalize(all_voxel_embeddings, p=2, dim=1)

    print(all_text_embeddings.min(), all_text_embeddings.max())
    print(all_voxel_embeddings.min(), all_voxel_embeddings.max())
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(all_text_embeddings, all_voxel_embeddings.T)
    print(similarity_matrix.min(), similarity_matrix.max())
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    # Ground truth indices
    ground_truth_indices = torch.arange(len(all_text_embeddings)).to(device)
    
    # 计算 top-k 匹配准确率
    hits = 0
    for i in range(len(ground_truth_indices)):
        # 如果 ground truth 出现在 top-k 索引中，则加 1
        if ground_truth_indices[i] in top_k_indices[i].tolist():
            hits += 1
            
    accuracy = hits / len(ground_truth_indices)

    return accuracy

# 设置随机种子
random.seed(22)
torch.manual_seed(22)

# 解析参数
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help='Dataset path')
    parser.add_argument('--voxel_folder', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress", help='Dataset path')
    parser.add_argument('--csv_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_npr.csv", help='Dataset path')
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Training dataset split ratio")
    parser.add_argument('--epochs', type=int, default=500, help="Training epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--temperature', type=float, default=0.1, help="Temperature")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()


# 主函数
def main():
    start_time = time.time()
    args = parsing()
    device = args.device

    # 加载数据集
    dataset = VoxelDataset1(csv_path=args.csv_path,
                                voxel_folder=args.voxel_folder,
                                use_tensor_condition=True
                                )
    # 按照split_ratio分割数据集
    train_size = int(args.split_ratio * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    vision_encoder = VisionEncoder(num_projection_layers=1, input_dims=512, projection_dims=256, dropout_rate=0.1).to(device)
    text_encoder = TextEncoder(num_projection_layers=1, input_dims=768, projection_dims=256, dropout_rate=0.1).to(device)
    dual_encoder = DualEncoder(text_encoder, vision_encoder, temperature=args.temperature).to(device)

    # vision_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/vision_encoder_new1.pth"))
    # vision_encoder.eval()  # 设置为评估模式
    # text_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/text_encoder_new1.pth"))
    # text_encoder.eval()  # 设置为评估模式
    # 定义优化器
    optimizer = optim.Adam(dual_encoder.parameters(), lr=0.001)
    # 监测 val_loss，当 patience 轮内无改善时，降低学习率 (factor=0.2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # 训练模型
    history = train_model(dual_encoder, train_loader, val_loader, scheduler, optimizer, num_epochs=args.epochs, device=device)

    # 保存模型
    # 保存图像编码器和文本编码器的模型
    torch.save(vision_encoder.state_dict(), "/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/vision_encoder_new3.pth")
    torch.save(text_encoder.state_dict(), "/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/text_encoder_new3.pth")

    torch.save(dual_encoder.state_dict(), "/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/dual_encoder_new3.pth")

    # 绘制训练和验证损失曲线
    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/training_results")
    
    
    # 训练准确率
    print("Scoring training data...")

    train_accuracy = compute_top_k_accuracy(train_loader, args.batch_size, vision_encoder,
                                            text_encoder)
    print(f"Train accuracy: {round(train_accuracy * 100, 3)}%")
    # 验证准确率
    print("Scoring evaluation data...")
    eval_accuracy = compute_top_k_accuracy(val_loader, args.batch_size, vision_encoder,
                                            text_encoder)
    print(f"Eval accuracy: {round(eval_accuracy * 100, 3)}%")
    
if __name__ == '__main__':
    main()