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

class VoxelDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, use_tensor_condition=False):

        self.dataset_path = dataset_folder
        self.transform = transform
        self.use_tensor_condition = use_tensor_condition
        
        # 加载体素数据
        self.voxels = np.load(os.path.join(dataset_folder, "geometries_2000x64x64x64.npy"))
        self.voxels = self.voxels.astype(np.float32)  # 确保数据类型为 float32
        
        # 读取图像路径和标题数据
        # self.images_dir = os.path.join(dataset_folder, "figs")
        annotation_file = os.path.join(dataset_folder, "captions.csv")
        self.annotations = pd.read_csv(annotation_file)["Captions"].tolist()
        
         # 生成独热编码标签
        self.labels = np.zeros((2000, 20), dtype=np.float32)  # (2000, 20)
        for i in range(20):
            for j in range(100):
                self.labels[i * 100 + j, i] = 1
        
        # # 创建图像-标题字典
        # self.image_path_to_caption = collections.defaultdict(list)
        # for i, caption in enumerate(self.annotations):
        #     image_path = os.path.join(self.images_dir, f"{i}")
        #     self.image_path_to_caption[image_path] = caption
        
        # self.image_paths = list(self.image_path_to_caption.keys())
        
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
        voxel = torch.tensor(self.voxels[idx])           # 转换为 PyTorch 张量   [64,64,64]
        # voxel = self.augment_voxel_tensor(voxel)
        voxel = 2 * voxel - 1                            # [0,1]->[-1,1]
        voxel = voxel.unsqueeze(0)
        # image_path = self.image_paths[idx]
        # image = self.read_image(image_path)
        caption = self.annotations[idx]  # 获取对应的文本描述
        res["occupancy"] = voxel                          
        res["caption"] = caption
        labels = torch.tensor(self.labels[idx])
        res["label"] = labels
        res["idx"] = idx
        
        if self.use_tensor_condition:
            prop = torch.tensor(self.prop[idx])           # 物理属性数据
            res["tensor_feature"]=prop
            # 物理属性拼接为文本
            p_phi = 0.7
            p_invE = 0.5
            p_aniso = 0.3

            # 每个属性是否保留
            keep_phi = random.random() < p_phi
            keep_invE = random.random() < p_invE
            keep_aniso = random.random() < p_aniso

            # 构造属性字符串（只拼接保留的）
            attr_parts = []
            if keep_phi:
                attr_parts.append(f"phi:{prop[0]:.4f}")
            if keep_invE:
                attr_parts.append(f"1/E:{prop[1]:.4f}")
            if keep_aniso:
                attr_parts.append(f"Anisotropy:{prop[2]:.4f}")

            # 若全部被mask掉，则填"None"
        
            prop_str = ", ".join(attr_parts) if attr_parts else "None"
            res["prop"] = prop_str
        else:
            res["prop"] = " "
        res["caption"] = caption+res["prop"]
        return res


class VoxelDataset1(Dataset):
    def __init__(self, csv_path, voxel_folder, use_tensor_condition=False, dim=(64,64,64)):
        """
        Args:
            csv_path: 包含 file_name 和 description 的 CSV 文件路径
            voxel_folder: 存放 .npy 体素文件的目录
        """
        self.df = pd.read_csv(csv_path)
        self.voxel_folder = voxel_folder
        self.use_tensor_condition = use_tensor_condition
        self.dim = dim
        # 提取文件名和描述
        self.file_names = self.df["file_name"].tolist()
        self.descriptions = self.df["description"].tolist()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        voxel_path = os.path.join(self.voxel_folder, self.file_names[idx])
        with open(voxel_path, 'rb') as f:
            voxel_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
            voxel = np.reshape(voxel_data, self.dim, order="F").astype(np.float32)
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
            E_data_map = scaler_E.transform(E_data).reshape(-1)

            v_data = tensor_feature[6:7].reshape(1, -1)    # 标准化C12泊松比相关项
            scaler_v = joblib.load("./scaler_C12")
            v_data_map = scaler_v.transform(v_data).reshape(-1)

            tensor_feature = np.array([E_data_map[0], v_data_map[0], binary_data_vol[0]], dtype=np.float32)

            # rand = random.random()
            # if rand < 0.5:
            #     prop_str = "None"
            p_phi = 0.7
            p_invE = 0.5
            p_aniso = 0.3

            # 每个属性是否保留
            keep_phi = random.random() < p_phi
            keep_invE = random.random() < p_invE
            keep_aniso = random.random() < p_aniso
            
            # 构造属性字符串（只拼接保留的）
            attr_parts = []
            if keep_phi:
                attr_parts.append(f"phi:{binary_data_vol[0]:.4f}")
            if keep_invE:
                attr_parts.append(f"E:{E_data_map[0]:.4f}")
            if keep_aniso:
                attr_parts.append(f"v:{v_data_map[0]:.4f}")

            # 若全部被mask掉，则填"None"
            prop_str = " ".join(attr_parts) if attr_parts else "None"
            caption = prop_str

        sample = {"occupancy": voxel, "caption": caption, "prop": tensor_feature}
        return sample