# 3D Microstructure Generation via Diffusion Model

## 项目概述
本项目实现了一个基于扩散模型的3D微观结构生成系统，支持**文本条件控制**和**物理属性条件控制**。模型能够根据材料描述文本和期望的物理属性（体积分数、弹性模量、泊松比等）生成对应的3D体素结构。
![模型结构图](figure\overview.png)

## 项目结构
Propdiff-TMG/
├── train.py # 模型训练入口
├── generate.py # 条件生成主程序
├── refinement.py # 奖励引导的迭代优化生成
├── chamfer.py # Chamfer距离计算
├── FID.py # FID、CLIP、分类准确率等评估指标
├── discriminator.py # 判别器（用于GAN-based评估/优化）
├── p_evl.py # 物理属性预测误差评估
├── figure.py # 结果可视化（散点图+拟合线）
├── temp.py # 临时评估脚本（奖励优化之后统一FID指标的数据划分）
├── network/ # 网络模块目录
│ ├── model_trainer.py # 扩散模型训练器
│ ├── model_utils.py # 网络基础模块
│ ├── model.py # 扩散模型
│ ├── classifier_net.py # 分类器
│ ├── dual_encoder.py # 视觉-文本双编码器
│ ├── data_loader_text.py # 文本条件数据加载
│ ├── solver.py # 属性预测器
│ └── unet.py # unet网络结构
├── utils/ # 工具函数
│ ├── mesh_utils.py # 体素转网格(obj)等
│ └── utils.py # 通用工具函数
└── data/ # 数据目录

## 环境配置
```bash
conda env create -f environment.yaml
conda activate microstructure
```

## 数据集
下载数据集并放在 `data/` 目录：
- [Geometries 2000](https://drive.google.com/drive/folders/1GFdJIUzeH-zgFM6HAgNifzAZXmKrmJs5)
- [GenText-Microstruct](https://drive.google.com/drive/folders/1fNj_v-8YjtYCPoyXn6qZ-HzG0LqAJeV9)

## 训练命令
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=0 python train.py --name debug/old --batch_size 4 --new True --continue_training True --image_size 64 --training_epoch 2000 --ema_rate 0.999 --base_channels 64  --save_last True --save_every_epoch 200 --with_attention True  --lr 2e-4 --optimizier adamw --verbose False --use_tensor_condition True
```

## 奖励优化命令
```bash
CUDA_VISIBLE_DEVICES=7 python refinement.py \
  --model_path /home/daibingxuan/workspace/microstructure_generation_3d/results/debug/textaddprop_aug/best-loss-epoch=1869-loss=0.1245.ckpt \
  --output_folder /home/daibingxuan/workspace/microstructure_generation_3d/evaluate \
  --dataset_path /home/daibingxuan/workspace/microstructure_generation_3d/data/datasets \
  --classifier_ckpt /home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier.pth \
  --batch_size 8 \
  --steps 50 \
  --num_generate 10 \
  --rounds 5 \
  --candidates 8 \
  --tensor_w 1.0 \
  --use_ema True
```
## 评估命令
```bash
python FID.py
```
## 推理命令
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_VISIBLE_DEVICES=7 python generate.py --model_path /home/daibingxuan/workspace/microstructure_generation_3d/results/debug/textaddprop_aug/best-loss-epoch=1869-loss=0.1245.ckpt --generate_method generate_based_on_text  --num_generate 10 --steps 100 --tensor_w 1 
```