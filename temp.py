import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from FID import calculate_fid
from network.classifier_net import Classifier
from refinement import threshold
import torch
from tqdm import tqdm
from network.data_loader_text import VoxelDataset1

# 加载
data = np.load('/home/daibingxuan/workspace/microstructure_generation_3d/evaluate/all_voxels_stand_newdata_nonorm.npy')  # 比如 'all_voxel.npy'
# voxels = np.load(os.path.join("/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", "geometries_2000x64x64x64.npy")).astype(np.float32)
# voxels = torch.tensor(voxels, dtype=torch.float32).to("cuda")  # 添加通道维度
# labels = np.zeros((2000, 20), dtype=np.float32)
# for i in range(20):
#     labels[i * 100:(i + 1) * 100, i] = 1
# index = np.arange(len(voxels))  # 获取所有数据的索引
# np.random.seed(0)  # 设置随机种子，确保结果可复现
# np.random.shuffle(index)  # 打乱索引
dataset = VoxelDataset1("/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_rand.csv", "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/rand8000", use_tensor_condition=True)
total_len = len(dataset)
num_samples = 2000
np.random.seed(0)
indices = np.random.permutation(total_len)[:num_samples]

voxels = []
all_captions = []
props = []
caption2voxel_tensor = {}
for idx in indices:
    # print(idx)
    res = dataset[idx]  
    voxels.append(res['occupancy'])
    all_captions.append(res['caption'])
    caption2voxel_tensor[res['caption']] = res['occupancy'].to("cuda")
    # props.append(res['prop'])

voxel = np.array(voxels).astype(np.float32)
voxels = torch.tensor(voxel, dtype=torch.float32).to("cuda")
# prop = np.array(props)
# data = data[:,0,:,:,:]
all_voxels_repeated = np.repeat(data, repeats=1, axis=0)

all_voxels=torch.tensor(all_voxels_repeated).to("cuda")

print("Loading Classifier...")
classifier = Classifier()

# Load the model weights into the classifier
classifier.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier.pth"))
classifier = classifier.to("cuda")
classifier.eval()


fid=calculate_fid(classifier, threshold(all_voxels.squeeze(1), 0), threshold(voxels.squeeze(1), 0), 8, "cuda")
print("fid:", fid)

# acc_score=0
# l=labels[index[:200]]
# # for idx, i in enumerate(tqdm(index[:200])):
# _, pred = classifier(threshold(all_voxels, 0.0))
# y_true = torch.tensor(l, dtype=torch.float32, device='cuda')
# acc = (pred.argmax(dim=1) == y_true.argmax(dim=1)).float().mean()
# acc_score+=acc

# print("acc:",acc)
# print("avry acc:",acc_score/200)