import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import str2bool
import numpy as np
import pandas as pd
from chamfer import calculate_chamfer_distance
import argparse
import sklearn
from scipy.linalg import sqrtm
from torch.utils.data import Subset, DataLoader, Dataset
from network.classifier_net import Classifier
from network.dual_encoder import VisionEncoder, TextEncoder
from generate import generate_based_on_text
from utils.mesh_utils import voxel2mesh
import joblib
torch.set_num_threads(2)

class VoxelDatasetClean(Dataset):
    def __init__(self, csv_path, voxel_folder, dim=(64,64,64)):
        self.df = pd.read_csv(csv_path)
        self.voxel_folder = voxel_folder
        self.dim = dim
        
        self.file_names = self.df["file_name"].tolist()
        self.descriptions = self.df["description"].tolist()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        voxel_path = os.path.join(self.voxel_folder, self.file_names[idx])
        
        # ---- 加载体素 ----
        with open(voxel_path, 'rb') as f:
            voxel_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
            voxel = np.reshape(voxel_data, self.dim, order="F").astype(np.float32)

        # 翻转修正
        for i in range(8):
            tmp = voxel[i*8:(i+1)*8, :, :]
            voxel[i*8:(i+1)*8, :, :] = tmp[::-1]

        voxel = torch.tensor(voxel).unsqueeze(0)  # [1,64,64,64]
        voxel = 2 * voxel - 1                     # [-1,1]

        caption = self.descriptions[idx]

        # ---- 加载材料属性 ----
        tensor_path = voxel_path.replace("binary_voxel", "binary_C")
        vol_path    = voxel_path.replace("binary_voxel", "vol")

        with open(tensor_path, 'rb') as f:
            binary_data = np.fromfile(f, dtype=np.float32)

        with open(vol_path, 'rb') as f:
            binary_data_vol = np.fromfile(f, dtype=np.float32)

        tensor_feature = - np.ones((10,), dtype=np.float32) 
        tensor_feature[0] = binary_data[0]     # x方向弹性模量
        tensor_feature[1] = binary_data[7]     # y方向弹性模量
        tensor_feature[2] = binary_data[14]    # z方向弹性模量
        tensor_feature[3] = binary_data[21]    # xy平面剪切模量
        tensor_feature[4] = binary_data[28]    # yz平面剪切模量
        tensor_feature[5] = binary_data[35]    # xz平面剪切模量
        tensor_feature[6] = binary_data[1]     # 泊松比
        tensor_feature[7] = binary_data[2]     # 其他耦合项
        tensor_feature[8] = binary_data[8]     # 其他耦合项

        E_data = tensor_feature[0:1].reshape(1, -1)    # 标准化C11弹性模量相关项
        scaler_E = joblib.load("./scaler_C11")
        # E_data_map = scaler_E.transform(E_data).reshape(-1)

        v_data = tensor_feature[6:7].reshape(1, -1)    # 标准化C12泊松比相关项
        scaler_v = joblib.load("./scaler_C12")
        # v_data_map = scaler_v.transform(v_data).reshape(-1)

        E_mapped = scaler_E.transform(E_data).reshape(-1)[0]
        v_mapped = scaler_v.transform(v_data).reshape(-1)[0]

        phi = binary_data_vol[0]

        tensor_feature = np.array([phi, E_mapped, v_mapped], dtype=np.float32)
        tensor_feature = torch.tensor(tensor_feature)  # [3]

        return {
            "occupancy": voxel,      # [1,64,64,64]
            "caption": caption,      # string
            "prop": tensor_feature   # [3]
        }

def cal_iou(real_voxels, fake_voxels, device):
    print(real_voxels.shape)
    print(fake_voxels.shape)
    real_voxels = real_voxels.to(device)
    fake_voxels = fake_voxels.to(device)
    # 计算交集和并集
    intersection = torch.sum(real_voxels * fake_voxels, dim=(1, 2, 3))  # 按照体素维度计算交集
    print(intersection)
     # 计算每个样本的体素数（非零数）
    real_count = torch.sum(real_voxels, dim=(1, 2, 3))
    fake_count = torch.sum(fake_voxels, dim=(1, 2, 3))
    print(real_count,fake_count)

    # 根号分母
    denominator = torch.sqrt(real_count * fake_count + 1e-6)
    print(denominator)
    # 相似度
    similarity = intersection / denominator

    return similarity.mean()

def calculate_clip(voxel, caption, prop_none, device="cuda"):
    vision_encoder = VisionEncoder(num_projection_layers=1, input_dims=512, projection_dims=256, dropout_rate=0.1).to(device)
    text_encoder = TextEncoder(num_projection_layers=1, input_dims=768, projection_dims=256, dropout_rate=0.1).to(device)
    vision_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/vision_encoder_new1.pth"))
    vision_encoder.eval()  
    text_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/text_encoder_new1.pth"))
    text_encoder.eval()
    # Compute embeddings
    _, voxel_embeddings = vision_encoder(voxel)
    print("prop: ", prop_none)
    text_embeddings = text_encoder([caption]*10, [prop_none]*10)
    # 复制文本嵌入 10 次，使其匹配 10 个体素
    # text_embeddings = text_embeddings.expand(10, -1)  # [1, D] → [10, D]

    text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
    voxel_embeddings = torch.nn.functional.normalize(voxel_embeddings, p=2, dim=1)

    # 计算相似度
    similarity_scores = torch.sum(text_embeddings * voxel_embeddings, dim=1)  # [10]
    # 计算 CLIP 分数
   
    clip_score = similarity_scores.mean().item()
    return clip_score

def calculate_fid(model, fake_data, real_data, batch_size, device="cuda"):
    model.eval()
    print("Fake Data Range:", fake_data.min(), fake_data.max())
    print("Real Data Range:", real_data.min(), real_data.max())

    generate_loader = DataLoader(fake_data, batch_size=batch_size, shuffle=False)
    real_loader = DataLoader(real_data, batch_size=batch_size, shuffle=False)

    model = VisionEncoder(num_projection_layers=1, input_dims=512, projection_dims=256, dropout_rate=0.1).to(device)
    model.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/vision_encoder_new1.pth"))
    model.eval()
    
    act1, act2 = [], []

    with torch.no_grad():
        # 计算生成数据的特征
        for x in generate_loader:
            x = x.unsqueeze(1).to(device)  # 确保通道维度
            embedding, _ = model(x)
            act1.append(embedding.cpu().numpy())

        # 计算真实数据的特征
        for x in real_loader:
            x = x.unsqueeze(1).to(device)
            embedding, _ = model(x)
            act2.append(embedding.cpu().numpy())

    # 拼接所有 batch
    act1 = np.concatenate(act1, axis=0).reshape(-1, 512)
    act2 = np.concatenate(act2, axis=0).reshape(-1, 512)
    print(act1.shape)
    print(act2.shape)
    # print("Real Data Mean:", act1.mean(axis=0)[:10])  # 只打印前 10 个
    # print("Generated Data Mean:", act2.mean(axis=0)[:10])
    # print("Real Data Variance:", np.var(act1, axis=0)[:10])
    # print("Generated Data Variance:", np.var(act2, axis=0)[:10])

    # 计算均值和协方差
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # 计算 FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real 

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# 解析命令行参数
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help='Dataset path')
    parser.add_argument('--model_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/results/debug/old/best-loss-epoch=455-loss=0.1303_1.ckpt", help='model path')
    parser.add_argument("--output_path", type=str, default="text_results/")
    parser.add_argument('--classifier_ckpt', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier.pth", help='classifier path')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--tensor_w", type=float, default=2)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()
    return args


# 主程序
def main():
    args = parsing()
    device = args.device
    # 数据集和 DataLoader
    # 读取体素数据
    dataset = VoxelDatasetClean("/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_rand.csv", "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/rand8000")
    total_len = len(dataset)
    num_samples = 2000
    np.random.seed(0)
    indices = np.random.permutation(total_len)[:num_samples]

    voxels = []
    annotations = []
    props = []

    for idx in indices:
        # print(idx)
        res = dataset[idx]  
        voxels.append(res['occupancy'])
        annotations.append(res['caption'])
        props.append(res['prop'])

    voxels = np.array(voxels).astype(np.float32)
    prop = np.array(props)
    # print(voxels.shape)
    # voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(np.float32)

    # # 读取文本标签
    # annotation_file = os.path.join(args.dataset_path, "captions.csv")
    # annotations = pd.read_csv(annotation_file)["Captions"].tolist()  # 取出标题

    # properties_file = os.path.join(args.dataset_path, "properties.csv")
    # properties = pd.read_csv(properties_file)
    # E_label = np.expand_dims(1 / properties["E"].values, axis=-1)  # 计算 1/E
    # Ani_label = np.expand_dims(properties["Anisotropy"].values, axis=-1)
    # phi = np.expand_dims(properties["Phi"].values, axis=-1)
    # prop = np.concatenate((phi, E_label, Ani_label), axis=-1).astype(np.float32)


    # # 生成独热编码标签 (2000, 20)
    # labels = np.zeros((2000, 20), dtype=np.float32)
    # for i in range(20):
    #     labels[i * 100:(i + 1) * 100, i] = 1
    
    
    print("Loading Classifier...")
    classifier = Classifier()

    # Load the model weights into the classifier
    classifier.load_state_dict(torch.load(args.classifier_ckpt))
    classifier = classifier.to(device)
    classifier.eval()


    # 打乱索引
    index = np.arange(2000)
    
    np.random.shuffle(index)

    # 处理体素数据
    voxels = torch.tensor(voxels, dtype=torch.float32).to(device)  # 添加通道维度
    voxels = (voxels + 1) / 2
    chamfer_scores = 0
    clip = 0
    total_sum = 0
    total_correct = 0
    generated_voxels_total = torch.empty(0, 64, 64, 64)

    for idx, i in enumerate(index[:200]):
        prop_str = f"This mechanical metamaterial. phi:{prop[i][0]:.4f} E:{prop[i][1]:.4f} v:{prop[i][2]:.4f}"
        prop_none = "phi:None, 1/E:None, Anisotropy:None."
        # 拼接到原始文本
        caption = annotations[i] + " " + prop_str 
        # caption = annotations[i] 
        prompt =("Task Objective: Generate a voxel structure based on the following description.\n\n" +
                "Material Description: " + caption + "\n\n" +
                "Physical Property: " + prop_none)
           
        # mesh = voxel2mesh(voxels[i].cpu().numpy())
        # mesh.export("real.obj")
        print("<<<<<caption:", idx, prompt)
        generated_voxels, generated_voxels_bt=generate_based_on_text(model_path=args.model_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                                 num_generate=args.num_generate, truncated_time=args.truncated_time,
                                 query=prop_str, prop=prop_none, w=args.tensor_w)
        # generated_voxels_norm = (generated_voxels - generated_voxels.min()) / (generated_voxels.max() - generated_voxels.min() + 1e-6)
        generated_voxels_total = torch.cat((generated_voxels_total, generated_voxels_bt), dim=0)
        # 计算iou
        # iou += cal_iou(voxels[i].repeat(10,1,1,1),generated_voxels_bt.to(device), device)
        chamfer_scores += calculate_chamfer_distance(generated_voxels_bt, voxels[i].repeat(10,1,1,1), args.device)
        generated_voxels_bt = generated_voxels_bt.unsqueeze(1).to(device)
        generated_voxels = generated_voxels.unsqueeze(1).to(device)
        # 计算clip
        clip += calculate_clip(generated_voxels, prop_str, prop_none)
        # # 计算分类准确率
        # _, pred = classifier(generated_voxels_bt)
        # y_true = torch.tensor(labels[i], dtype=torch.float32, device=device).repeat(10, 1)
        # acc = (pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean()
        
        total_sum += 1
        # total_correct += acc.item()
        # t = total_correct / total_sum
        chamfer_score = chamfer_scores / total_sum
        clip_score = clip / total_sum
        print(f"chamfer_score: {chamfer_score:.4f}")
        print(f"clip_score: {clip_score:.4f}")
        # print(f"epoch: {idx}, Accuracy: {t:.4f}, ACC: {acc.item():.4f}")

    # # 计算最终准确率
    # final_acc = total_correct / total_sum
    # print(f"Accuracy: {final_acc:.4f}")
    print(generated_voxels_total.shape, voxels.shape)
    # 计算 FID
    fid_score = calculate_fid(classifier, generated_voxels_total.squeeze(1), voxels.squeeze(1), args.batch_size, device)
    print(f"FID Score: {fid_score:.4f}")

    
    # 保存结果到txt
    output_metrics_path = os.path.join(args.output_path, "metrics_new.txt")
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_metrics_path, "w") as f:
        f.write(f"chamfer Score: {chamfer_score:.4f}\n")
        f.write(f"CLIP Score: {clip_score:.4f}\n")
        # f.write(f"Accuracy: {final_acc:.4f}\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
    print(f"Metrics saved to {output_metrics_path}")

if __name__ == '__main__':
    main()
