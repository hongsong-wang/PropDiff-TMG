import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import numpy as np
from network.model_trainer import DiffusionModel
from network.dual_encoder import VisionEncoder, TextEncoder
from network.data_loader_text import VoxelDataset, VoxelDataset1
from network.classifier_net import Classifier
from chamfer import calculate_chamfer_distance
from discriminator import VoxelDiscriminator, PatchVoxelDiscriminator
from FID import calculate_fid
from utils.mesh_utils import voxel2mesh
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse

from tqdm import tqdm
import joblib
import math
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import time
from bitstring import BitArray
import pdb

def threshold(voxels: torch.Tensor, th=0.0) -> torch.Tensor:
    """连续体素 [-1,1] → 二值体素 {0,1}"""
    return (voxels > th).float()


class RewardCal:
    def __init__(self,
                 classifier: None,
                 metrics_name: str,
                 metrics_list: str,
                 caption2voxel_map: dict,
                 device,
                 run_name="",
                 pdb_save_path="sc_tmp",
                 ss_match='a'):
        self.metrics_name = metrics_name.split(",")
        metrics_list = metrics_list.split(",")
        self.metrics_list = [float(x) for x in metrics_list]  # 权重
        assert len(self.metrics_name) == len(self.metrics_list)
        self.classifier = classifier
        self.fast_clip = FastClipScorer(device)
        self.discriminator = PatchVoxelDiscriminator().to(device)
        self.discriminator.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/discriminator_epoch_new.pth"))
        self.discriminator.eval()
        self.caption2voxel_map = caption2voxel_map
        self.device = device


    def __call__(self, voxels: torch.Tensor, captions: list[str]) -> torch.Tensor:

        bin_vox = threshold(voxels, th=0.0)  # 二值化体素
        
        # CLIP 得分
        clip_scores = self.fast_clip(voxels, captions, prop=None)  # (B,)
        
        # D_score得分
        # pred = self.discriminator(bin_vox)
        # d_score =  torch.sigmoid(pred).squeeze(-1)
        patch_pred, global_pred, final_logit = self.discriminator(bin_vox)
        d_score_patch = torch.sigmoid(patch_pred).mean(dim=[2,3,4]).squeeze(1)
        d_score_global = torch.sigmoid(global_pred).squeeze()  # global_pred 通常是 (B, 1)，去掉1维
        d_score_final = final_logit
        d_score = 0.4 * d_score_patch + 0.4 * d_score_global + 0.2 * d_score_final
        print("clip_scores",clip_scores)
        print("d_scores",d_score)
        # 结合权重进行加权
        total_score = (
            self.metrics_list[0] * clip_scores +
            self.metrics_list[1] * d_score
        )
        # # 拼接
        # rewards = torch.stack([clip_scores, d_score], dim=1)  # (B, 2)
        # weights = torch.tensor(self.metrics_list, device=self.device)     # (2,)

        # # 优势函数归一（Advantage Normalization）
        # advantages = rewards - rewards.mean(dim=0, keepdim=True)
        # normalized_adv = advantages / (rewards.std(dim=0, keepdim=True) + 1e-6)

        # total_score = (normalized_adv * weights).sum(dim=1)  # (B,)

        return clip_scores  # 返回最终的加权分数
    
    def evaluate_metrics(self, voxels: torch.Tensor, captions: list[str]):

        bin_vox = threshold(voxels, th=0.0)
        reference_voxels = threshold(reference_voxels_for_text(captions, self.caption2voxel_map), th=0.0)
        chamfer_scores = calculate_chamfer_distance(bin_vox, reference_voxels, self.device)
        clip_scores = self.fast_clip(voxels, captions,prop=None)
        # pred = self.discriminator(bin_vox)
        # d_score =  torch.sigmoid(pred).squeeze(-1)
        patch_pred, global_pred, final_logit = self.discriminator(bin_vox)
        d_score_patch = torch.sigmoid(patch_pred).mean(dim=[2,3,4]).squeeze(1)
        d_score_global = torch.sigmoid(global_pred).squeeze()  # global_pred 通常是 (B, 1)，去掉1维
        d_score_final = torch.sigmoid(final_logit)
        d_score = 0.4 * d_score_patch + 0.4 * d_score_global + 0.2 * d_score_final
        print(bin_vox.shape, reference_voxels.shape)
        fid_scores = calculate_fid(self.classifier, bin_vox.squeeze(1), reference_voxels.squeeze(1), batch_size=bin_vox.shape[0], device=self.device)
        return chamfer_scores, clip_scores, d_score, fid_scores


def reference_voxels_for_text(captions: list[str], caption2voxel_tensor: dict) -> torch.Tensor:
    refs = []
    for caption in captions:
        if caption not in caption2voxel_tensor:
            raise KeyError(f"Caption not found in dataset: '{caption}'")
        refs.append(caption2voxel_tensor[caption].unsqueeze(0))  # [1, 1, 64, 64, 64]
    return torch.cat(refs, dim=0)  # [B, 1, 64, 64, 64]


# def compute_iou(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
#     # pred, ref: B×1×D×H×W（0/1）
#     inter =   (pred * ref).sum(dim=(1,2,3,4))                   # (B,)
#     union = ((pred + ref) > 0).sum(dim=(1,2,3,4)).clamp(min=1)  # (B,)
#     return inter / union

class FastClipScorer:
    def __init__(self, device="cuda"):
        self.device = device
        self.vision_encoder = VisionEncoder(num_projection_layers=1, input_dims=512, projection_dims=256, dropout_rate=0.1).to(device)
        self.text_encoder = TextEncoder(num_projection_layers=1, input_dims=768, projection_dims=256, dropout_rate=0.1).to(device)
        self.vision_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/vision_encoder_new1.pth"))
        self.text_encoder.load_state_dict(torch.load("/home/daibingxuan/workspace/microstructure_generation_3d/training_result/dual_encoder/text_encoder_new1.pth"))
        self.vision_encoder.eval()
        self.text_encoder.eval()

    def __call__(self, voxel, caption_list, prop):
        with torch.no_grad():
            _, voxel_emb = self.vision_encoder(voxel)

            # # 保证 caption_list 是 N 个文本，而不是 1 个文本 expand
            # if isinstance(caption_list, str):
            #     caption_list = [caption_list] * voxel_emb.shape[0]
            # if isinstance(prop, str):
            #     prop = [prop] * voxel_emb.shape[0]

            text_emb = self.text_encoder(caption_list, prop)

            voxel_emb = torch.nn.functional.normalize(voxel_emb, p=2, dim=1)
            text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
            return (voxel_emb * text_emb).sum(dim=1)


def generate_with_reward(
    model_path: str,
    query: str,
    ema: bool = True,
    reward_model=None,
    num_generate: int =10,
    truncated_time: float = 0.0,
    steps: int = 50,
    edit_num: int = 4,
    rounds: int = 5,                   # 编辑轮数
    K: int = 8,                        # 每轮生成 K 个候选
    w: float = 1.0
) -> torch.Tensor:

    # 初次完整采样
    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    time_start = time.time()
    vox = generator.sample_with_text(caption=query, prop=None, batch_size=num_generate,
                                                steps=steps, truncated_index=truncated_time, tensor_w=w)
    time_end = time.time()
    print('generte time cost', time_end - time_start, 's')

    B, _, D, H, W = vox.shape
    best_vox = vox.clone()  # 历史最佳体素
    with torch.no_grad():
        best_reward = reward_model(vox, [query]*B)  # 初始 reward

        for rnd in tqdm(range(rounds), desc='Optimization Rounds'):
            # 局部生成位置
            loc_set = [
                [  # 每个batch独立
                    (torch.randint(0, D-16, (1,)).item(),
                    torch.randint(0, H-16, (1,)).item(),
                    torch.randint(0, W-16, (1,)).item())
                    for _ in range(edit_num)
                ]
                for _ in range(B)
            ]

            for edit_idx in range(edit_num):
                candidates = []
                for k in range(K):
                    cand = vox.clone()

                    for b in range(B):
                        d0, h0, w0 = loc_set[b][edit_idx]

                        # 重采样一份新的
                        patch = generator.sample_with_text(
                            caption=query,
                            prop=None,
                            batch_size=1,
                            steps=steps//2,
                            truncated_index=0.0,
                            tensor_w=w  # or something needed
                        )[0:1]  # shape (1,1,D,H,W)

                        # 局部替换
                        cand[b:b+1, :, d0:d0+16, h0:h0+16, w0:w0+16] = patch[:, :, d0:d0+16, h0:h0+16, w0:w0+16]

                    candidates.append(cand)

                # (candidates_per_edit, B, 1, D, H, W) → (B, candidates_per_edit, 1, D, H, W)
                all_cands = torch.stack(candidates, dim=0).permute(1, 0, 2, 3, 4, 5)

                # 按batch单独挑最优
                new_vox_list = []
                for b in range(B):
                    cands_b = all_cands[b]  # (candidates_per_edit, 1, D, H, W)
                    rewards = reward_model(cands_b, [query]*K)  # (candidates_per_edit,)
                    best_idx = rewards.argmax()
                    new_vox_list.append(cands_b[best_idx])

                vox = torch.stack(new_vox_list, dim=0)  # (B,1,D,H,W)

            # 这一轮完成，评估一下reward
            rewards_now = reward_model(vox, [query]*B)  # (B,)

            # 更新历史最佳
            improved = rewards_now > best_reward
            best_vox[improved] = vox[improved]
            best_reward[improved] = rewards_now[improved]

            # softmax一把reward，作为重采样权重
            reward_weights = torch.softmax(best_reward * 5.0, dim=0)  # 温度可调
            sampled_idx = torch.multinomial(reward_weights, B, replacement=True)

            # 重采样
            vox = best_vox[sampled_idx]

    return best_vox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxel structure generation with reward-guided sampling")

    parser.add_argument("--model_path", type=str, default="/home/daibingxuan/workspace/microstructure_generation_3d/results/debug/textaddprop_aug/best-loss-epoch=1869-loss=0.1245.ckpt",help="Path to the trained diffusion model checkpoint")
    parser.add_argument("--output_folder", type=str, default="/home/daibingxuan/workspace/microstructure_generation_3d/evaluate", help="Path to the output folder")
    parser.add_argument("--dataset_path", type=str, default="/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets", help="Path to the voxel dataset folder")
    parser.add_argument('--classifier_ckpt', type=str, default=r"/home/daibingxuan/workspace/microstructure_generation_3d/training_result/classifier.pth", help='classifier path')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--num_generate", type=int, default=1, help="Number of generate")
    parser.add_argument("--rounds", type=int, default=5, help="Number of editing rounds")
    parser.add_argument("--candidates", type=int, default=4, help="Number of candidates per round")
    parser.add_argument("--truncated_time", type=float, default=0.0, help="Truncation index for partial sampling")
    parser.add_argument("--tensor_w", type=float, default=1.0, help="Weight parameter for the diffusion model")
    parser.add_argument("--use_ema", type=str2bool, default=True, help="Use EMA model or not")
  
    args = parser.parse_args()
    # 加载数据集
    # dataset = VoxelDataset(dataset_folder=args.dataset_path, use_tensor_condition=False)
    # # 文本-体素的映射
    # caption2voxel_tensor = {}
    # all_captions = []
    # for data in dataset:
    #     caption = data["caption"]
    #     voxel = data["occupancy"]
    #     caption2voxel_tensor[caption] = voxel.to("cuda")
    #     all_captions.append(caption)
    
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
        props.append(res['prop'])

    voxel = np.array(voxels).astype(np.float32)
    prop = np.array(props)

    print("Loading Classifier...")
    classifier = Classifier()

    # Load the model weights into the classifier
    classifier.load_state_dict(torch.load(args.classifier_ckpt))
    classifier = classifier.to("cuda")
    classifier.eval()
    # 打乱索引
    index = np.arange(len(all_captions))  # 获取所有数据的索引
      # 设置随机种子，确保结果可复现
    np.random.shuffle(index)  # 打乱索引

    # 奖励函数的定义
    reward_model = RewardCal(classifier=classifier, metrics_name="clip,d_score", metrics_list="1.0,0.4", caption2voxel_map=caption2voxel_tensor, device="cuda")
    all_voxels = []  # 用于收集所有样本
    all_chamfer, all_clip, all_fid = [], [], []
    # 用于保存最优生成
    best_voxels = []  
    gt_voxels = []    
    import time

    start_time = time.time()
    txt_log = os.path.join(args.output_folder, "test.txt")
    with open(txt_log, "w") as f:
        for idx, i in enumerate(tqdm(index[:1], desc="Generating voxels for selected captions")):
            caption = all_captions[i]
            gt_voxels.append(caption2voxel_tensor[caption].cpu().numpy())
            with torch.no_grad():
                output_voxel = generate_with_reward(
                    model_path=args.model_path,
                    query=caption,
                    ema=args.use_ema,
                    reward_model=reward_model,
                    num_generate=args.num_generate,
                    truncated_time=args.truncated_time,
                    steps=args.steps,
                    rounds=args.rounds,
                    K=args.candidates,
                    w=args.tensor_w
                )
                end_time = time.time()
                print(f"Time cost: {end_time - start_time:.4f} s")
                npy_vox = output_voxel.cpu().squeeze(1).numpy()
                # for jj in range(args.num_generate):
                #     voxel = npy_vox[jj]
                #     voxel[voxel>0] = 1
                #     voxel[voxel<0] = 0
                #     # save to obj
                #     try:
                #         mesh = voxel2mesh(voxel)
                #         mesh.export(os.path.join("/home/daibingxuan/workspace/microstructure_generation_3d/text_results/text1", str(idx)+'_'+str(jj) + ".obj"))
                        
                #     except Exception as e:
                #         print(str(e))

                all_voxels.append(npy_vox)

                # 评估指标
                chamfer_score, clip_score, d_score, fid_score = reward_model.evaluate_metrics(output_voxel, [caption]*output_voxel.shape[0])
                rewards = reward_model(output_voxel, [caption]*output_voxel.shape[0])
            best_idx = rewards.argmax().item()
            best_voxel = output_voxel[best_idx]
            
            best_chamfer = chamfer_score
            best_clip = clip_score[best_idx].item()
            best_fid = fid_score.item()
            
            best_voxels.append(best_voxel.cpu().numpy())
            
            all_chamfer.append(best_chamfer)
            all_clip.append(best_clip)
            all_fid.append(best_fid)

            # 写入当前样本的指标
            f.write(f"Sample {idx}: caption={caption}\n")
            f.write(f"   best IoU: {best_chamfer:.4f}, best CLIP: {best_clip:.4f}, best FID: {best_fid:.4f}\n\n")
            f.flush()

         # 保存所有 best 生成体素
        best_voxels_array = np.stack(best_voxels, axis=0)
        gt_voxels_array = np.stack(gt_voxels, axis=0)
        # 写入总平均
        f.write("\n======== Overall Metrics ========\n")
        f.write(f"Overall Avg IoU: {np.mean(all_chamfer):.4f}\n")
        f.write(f"Overall Avg CLIP: {np.mean(all_clip):.4f}\n")
        f.write(f"Overall Avg FID: {np.mean(all_fid):.4f}\n")
        print("Calculating FID between generated voxels and ground truth...")
        gen_voxel_tensor = torch.from_numpy(best_voxels_array).squeeze(1).to("cuda").float()
        gt_voxel_tensor = torch.from_numpy(gt_voxels_array).squeeze(1).to("cuda").float()
        # fid_scores = calculate_fid(classifier, threshold(gen_voxel_tensor, th=0.0), threshold(gt_voxel_tensor, th=0.0), batch_size=gen_voxel_tensor.shape[0], device="cuda")
        # print("fid:",fid_scores)
        # f.write(f"Overall FID: {fid_scores:.4f}\n")
        f.flush()
    
    all_voxels_array = np.concatenate(all_voxels, axis=0)
    np.save(os.path.join(args.output_folder, "all_voxels_stand_newdata_nonorm1.npy"), all_voxels_array)
    np.save(os.path.join(args.output_folder, "all_voxels_best_newdata_nonorm1.npy"), best_voxels_array)
    print(f"Saved all voxel samples to {args.output_folder}, shape = {all_voxels_array.shape}")
    print(f"Saved all voxel samples to {args.output_folder}, shape = {best_voxels_array.shape}")
    fid_scores = calculate_fid(classifier, threshold(gen_voxel_tensor, th=0.0), threshold(gt_voxel_tensor, th=0.0), batch_size=gen_voxel_tensor.shape[0], device="cuda")
    print("fid:",fid_scores)
    # f.write(f"Overall FID: {fid_scores:.4f}\n")