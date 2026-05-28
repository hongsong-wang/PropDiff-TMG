import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import str2bool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sklearn
from scipy.linalg import sqrtm
from network.solver import Slover
from generate import generate_based_on_text
from utils.mesh_utils import voxel2mesh
from network.data_loader_text import VoxelDataset1

torch.set_num_threads(2)

# 解析命令行参数
def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=r"/home/daibingxuan/workspace/material_generation_LLM/data/datasets", help='Dataset path')
    parser.add_argument('--model_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/results/debug/new8/best-loss-epoch=305-loss=0.1566.ckpt", help='model path')
    parser.add_argument('--voxel_folder', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/rand8000", help='Dataset path')
    parser.add_argument('--csv_path', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_rand.csv", help='Dataset path')
    parser.add_argument("--output_path", type=str, default="text_results/")
    parser.add_argument('--solver_ckpt', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/training_result/solver/ckpt/solver_newdata_best.pth", help='classifier path')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
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
    voxels = np.load(os.path.join(args.dataset_path, "geometries_2000x64x64x64.npy")).astype(np.float32)

    # 读取文本标签
    annotation_file = os.path.join(args.dataset_path, "captions.csv")
    annotations = pd.read_csv(annotation_file)["Captions"].tolist()  # 取出标题

    properties_file = os.path.join(args.dataset_path, "properties.csv")
    properties = pd.read_csv(properties_file)
    E_label = np.expand_dims(1 / properties["E"].values, axis=-1)  # 计算 1/E
    Ani_label = np.expand_dims(properties["Anisotropy"].values, axis=-1)
    phi = np.expand_dims(properties["Phi"].values, axis=-1)
    phi_max, phi_min = phi.max(), phi.min()
    E_max, E_min = E_label.max(), E_label.min()
    Ani_max, Ani_min = Ani_label.max(), Ani_label.min()

    prop = np.concatenate((phi, E_label, Ani_label), axis=-1).astype(np.float32)
    # dataset = VoxelDataset1(csv_path=args.csv_path,
    #                             voxel_folder=args.voxel_folder,
    #                             use_tensor_condition=True
    #                             )
    # total_len = len(dataset)
    # num_samples = 2000
    # np.random.seed(0)
    # indices = np.random.permutation(total_len)[:num_samples]

    voxels = []
    annotations = []
    props = []

    # for idx in indices:
    #     # print(idx)
    #     res = dataset[idx]  
    #     voxels.append(res['occupancy'])
    #     annotations.append(res['caption'])
    #     props.append(res['prop'])

    # voxels = np.array(voxels).astype(np.float32)
    # prop = np.array(props)
    print("Loading Solver...")
    solver = Slover()
    data = np.load('/home/daibingxuan/workspace/microstructure_generation_3d/evaluate/all_voxels_best_newdata.npy') 
    # all_voxels_repeated = np.repeat(data, repeats=5, axis=0)

    # all_voxels=torch.tensor(all_voxels_repeated).to("cuda")
    # Load the model weights into the classifier
    solver.load_state_dict(torch.load(args.solver_ckpt))
    solver = solver.to(device)
    solver.eval()


    # 打乱索引
    index = np.arange(2000)
    np.random.seed(0)
    np.random.shuffle(index)

    # 处理体素数据
    data = torch.tensor(data, dtype=torch.float32).to(device).unsqueeze(1)  # 添加通道维度
    
    data=data*2-1
    error = 0
    total_sum = 0
    all_sample_errors = []
    all_predictions = []

    for idx, i in enumerate(index[:200]):
    # for i in range(2000):
        # mesh = voxel2mesh(voxels[i].cpu().numpy())
        # mesh.export("real.obj")
        # prop_str = f"phi:{prop[i][0]:.4f}, 1/E:{prop[i][1]:.4f}, Anisotropy:{prop[i][2]:.4f}."
        # 拼接到原始文本
        caption = annotations[i] 
        print("<<<<<caption:", caption)
        generated_voxels, generated_voxels_bt=generate_based_on_text(model_path=args.model_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                                 num_generate=args.num_generate, truncated_time=args.truncated_time,
                                 query=caption, prop=None, w=args.tensor_w)
        
        pred = solver(generated_voxels.unsqueeze(1).to(device))
        # pred = solver(data[idx])
        pred = pred.squeeze()
        all_predictions.append(pred.detach().cpu().numpy())
        # 归一化的误差
        error_norm = torch.zeros_like(pred)
        error_norm[0] = torch.abs((pred[0] - prop[i, 0]) / (phi_max - phi_min + 1e-8))
        error_norm[1] = torch.abs((pred[1] - prop[i, 1]) / (E_max - E_min + 1e-8))
        error_norm[2] = torch.abs((pred[2] - prop[i, 2]) / (Ani_max - Ani_min + 1e-8))
        

        mean_error = torch.mean(error_norm).item()
        all_sample_errors.append(round(mean_error, 2))

        error += mean_error
        total_sum += 1
        error_score = error / total_sum
        print(f"epoch: {i}, error: {mean_error:.4f}, avr_error: {error_score:.4f}")

    # 画误差直方图
    plt.figure(figsize=(8,6))
    plt.hist(all_sample_errors, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Error Value')
    plt.ylabel('Count')
    plt.title('Histogram of Average Prediction Errors')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_path, 'error_histogram_ref.png'))
    plt.close()
    # 计算最终准确率
    final_error = error / total_sum
    print(f"error: {final_error:.4f}")
    
    # 保存结果到txt
    output_metrics_path = os.path.join(args.output_path, "error_ref_new.txt")
    os.makedirs(args.output_path, exist_ok=True)
    with open(output_metrics_path, "w") as f:
        f.write(f"error: {final_error:.4f}")
    print(f"Metrics saved to {output_metrics_path}")
    
    pred_array = np.array(all_predictions)  # shape: (200, 3)
    pred_df = pd.DataFrame(pred_array, columns=["Predicted_Phi", "Predicted_1/E", "Predicted_Anisotropy"])
    csv_path = os.path.join(args.output_path, "predicted_results_ref_new.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"Prediction results saved to {csv_path}")

if __name__ == '__main__':
    main()
