import os
import numpy as np
import pandas as pd
import collections
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy.ndimage import rotate
import random
import joblib


# ============================================================
# VoxelDataset —— 主数据集：Geometries 2000
#   - geometries_2000x64x64x64.npy  (2000, 64, 64, 64)
#   - captions.csv                   文本描述（Captions 列）
#   - properties.csv                 物理属性（E, Anisotropy, Phi 列）
# ============================================================
class VoxelDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, use_tensor_condition=False):

        self.dataset_path = dataset_folder
        self.transform = transform
        self.use_tensor_condition = use_tensor_condition

        # 加载体素数据
        self.voxels = np.load(os.path.join(dataset_folder, "geometries_2000x64x64x64.npy"))
        self.voxels = self.voxels.astype(np.float32)

        # 读取文本标注
        annotation_file = os.path.join(dataset_folder, "captions.csv")
        self.annotations = pd.read_csv(annotation_file)["Captions"].tolist()

        # 生成独热编码标签（20 类 × 100 样本 = 2000）
        self.labels = np.zeros((2000, 20), dtype=np.float32)
        for i in range(20):
            for j in range(100):
                self.labels[i * 100 + j, i] = 1

        # 如果启用 use_tensor_condition，则加载 properties.csv
        if self.use_tensor_condition:
            properties_file = os.path.join(dataset_folder, "properties.csv")
            properties = pd.read_csv(properties_file)
            E_label = np.expand_dims(1 / properties["E"].values, axis=-1)  # 计算 1/E
            Ani_label = np.expand_dims(properties["Anisotropy"].values, axis=-1)
            phi = np.expand_dims(properties["Phi"].values, axis=-1)
            self.prop = np.concatenate((phi, E_label, Ani_label), axis=-1).astype(np.float32)
        else:
            self.prop = None

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        res = {}
        voxel = torch.tensor(self.voxels[idx])           # [64,64,64]
        voxel = 2 * voxel - 1                            # [0,1]->[-1,1]
        voxel = voxel.unsqueeze(0)                       # [1,64,64,64]

        caption = self.annotations[idx]
        res["occupancy"] = voxel
        res["caption"] = caption
        res["label"] = torch.tensor(self.labels[idx])
        res["idx"] = idx

        if self.use_tensor_condition:
            prop = torch.tensor(self.prop[idx])
            res["tensor_feature"] = prop
            # 物理属性拼接为文本（随机 dropout）
            p_phi = 0.7
            p_invE = 0.5
            p_aniso = 0.3

            keep_phi = random.random() < p_phi
            keep_invE = random.random() < p_invE
            keep_aniso = random.random() < p_aniso

            attr_parts = []
            if keep_phi:
                attr_parts.append(f"phi:{prop[0]:.4f}")
            if keep_invE:
                attr_parts.append(f"1/E:{prop[1]:.4f}")
            if keep_aniso:
                attr_parts.append(f"Anisotropy:{prop[2]:.4f}")

            prop_str = ", ".join(attr_parts) if attr_parts else "None"
            res["prop"] = prop_str
        else:
            res["prop"] = " "
            res["tensor_feature"] = torch.zeros(0)

        res["caption"] = caption + res["prop"]
        return res


# ============================================================
# VoxelDataset1 —— 备选数据集：GenText-Microstruct
#   - rand8000/ 或 randombulk_compress/  体素文件目录
#     每个样本包含：{id}binary_voxel, {id}binary_C, {id}vol
#   - voxel_descriptions_npr.csv  或  voxel_descriptions_rand.csv
#     列：file_name, description, E, V, G, bulk, num_components, has_voids
#
# 使用方式（在 train.py 中切换）：
#   csv_path = ".../voxel_descriptions_npr.csv"
#   voxel_folder = ".../randombulk_compress"
#   use_tensor_condition = True
# ============================================================
class VoxelDataset1(Dataset):
    def __init__(self, csv_path, voxel_folder, use_tensor_condition=False, dim=(64, 64, 64)):
        """
        Args:
            csv_path: 包含 file_name 和 description 的 CSV 文件路径
            voxel_folder: 存放 binary_voxel / binary_C / vol 文件的目录
            use_tensor_condition: 是否加载物理属性张量
            dim: 体素维度
        """
        self.df = pd.read_csv(csv_path)
        self.voxel_folder = voxel_folder
        self.use_tensor_condition = use_tensor_condition
        self.dim = dim
        self.file_names = self.df["file_name"].tolist()
        self.descriptions = self.df["description"].tolist()

        # 预加载 scaler（避免每个 __getitem__ 重复读取）
        if self.use_tensor_condition:
            # scaler 文件位于项目根目录（PropDiff-TMG/）
            _scaler_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self._scaler_E = joblib.load(os.path.join(_scaler_dir, "scaler_C11"))
            self._scaler_v = joblib.load(os.path.join(_scaler_dir, "scaler_C12"))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        voxel_path = os.path.join(self.voxel_folder, self.file_names[idx])
        with open(voxel_path, 'rb') as f:
            voxel_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
            voxel = np.reshape(voxel_data, self.dim, order="F").astype(np.float32)

        # 每 8 行翻转（与 voxeltotext.py 修复后保持一致）
        for i in range(8):
            tmp = voxel[i * 8:(i + 1) * 8, :, :]
            voxel[i * 8:(i + 1) * 8, :, :] = tmp[::-1]

        voxel = torch.tensor(voxel).unsqueeze(0)  # [1,D,H,W]
        voxel = 2 * voxel - 1

        caption = self.descriptions[idx]

        if self.use_tensor_condition:
            tensor_path = str(voxel_path).replace("binary_voxel", "binary_C")
            with open(tensor_path, 'rb') as f:
                binary_data = np.fromfile(f, dtype=np.float32)
            vol_path = str(voxel_path).replace("binary_voxel", "vol")
            with open(vol_path, 'rb') as f:
                binary_data_vol = np.fromfile(f, dtype=np.float32)

            tensor_feature = -np.ones((10,), dtype=np.float32)
            tensor_feature[0] = binary_data[0]      # x方向弹性模量
            tensor_feature[1] = binary_data[7]      # y方向弹性模量
            tensor_feature[2] = binary_data[14]     # z方向弹性模量
            tensor_feature[3] = binary_data[21]     # xy平面剪切模量
            tensor_feature[4] = binary_data[28]     # yz平面剪切模量
            tensor_feature[5] = binary_data[35]     # xz平面剪切模量
            tensor_feature[6] = binary_data[1]      # 泊松比
            tensor_feature[7] = binary_data[2]      # 其他耦合项
            tensor_feature[8] = binary_data[8]      # 其他耦合项

            E_data = tensor_feature[0:1].reshape(1, -1)
            E_data_map = self._scaler_E.transform(E_data).reshape(-1)

            v_data = tensor_feature[6:7].reshape(1, -1)
            v_data_map = self._scaler_v.transform(v_data).reshape(-1)

            # 3 维属性向量：[E_norm, v_norm, volume_fraction]
            tensor_feature = np.array(
                [E_data_map[0], v_data_map[0], binary_data_vol[0]],
                dtype=np.float32
            )

            # 属性文本拼接（随机 dropout）
            p_phi = 0.7
            p_invE = 0.5
            p_aniso = 0.3

            keep_phi = random.random() < p_phi
            keep_invE = random.random() < p_invE
            keep_aniso = random.random() < p_aniso

            attr_parts = []
            if keep_phi:
                attr_parts.append(f"phi:{binary_data_vol[0]:.4f}")
            if keep_invE:
                attr_parts.append(f"E:{E_data_map[0]:.4f}")
            if keep_aniso:
                attr_parts.append(f"v:{v_data_map[0]:.4f}")

            prop_str = " " + " ".join(attr_parts) if attr_parts else ""
            caption = caption + prop_str  # 文本描述 + 属性条件拼接

        # 统一返回键名：tensor_feature（与 VoxelDataset 一致）
        sample = {
            "occupancy": voxel,
            "caption": caption,
            "tensor_feature": torch.tensor(tensor_feature) if self.use_tensor_condition else torch.zeros(0),
        }
        return sample
