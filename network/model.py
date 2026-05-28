import torch
import torch.nn.functional as F
from tqdm import tqdm
from network.model_utils import *
from network.unet import UNetModel
from einops import rearrange, repeat
import numpy as np
from random import random
from functools import partial
from torch import nn
from torch.special import expm1
import sys
import joblib
import pdb

import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt


TRUNCATED_TIME = 0.7


class Propdiff(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            base_channels: int = 128,
            attention_resolutions: str = "16,8",
            with_attention: bool = False,
            num_heads: int = 4,
            dropout: float = 0.0,
            verbose: bool = False,
            use_text_condition: bool = True,
            use_tensor_condition: bool = False,
            eps: float = 1e-6,
            noise_schedule: str = "linear",
    ):
        super().__init__()
        self.image_size = image_size
        if image_size == 8:
            channel_mult = (1, 4, 8)
        elif image_size == 32:
            channel_mult = (1, 2, 4, 8)
        elif image_size == 64:
            channel_mult = (1, 2, 4, 8, 8)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
        self.eps = eps
        self.verbose = verbose
        self.use_text_condition = use_text_condition
        self.use_tensor_condition = use_tensor_condition
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        self.denoise_fn = UNetModel(
            image_size=image_size,
            base_channels=base_channels,
            dim_mults=channel_mult, dropout=dropout,
            use_text_condition=use_text_condition,
            use_tensor_condition=use_tensor_condition,
            world_dims=3,
            num_heads=num_heads,
            attention_resolutions=tuple(attention_ds), with_attention=with_attention,
            verbose=verbose)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        # self.perceptual_loss_fn = PerceptualLoss3D(
        #     device="cuda"
        # )

    @property
    def device(self):
        return next(self.denoise_fn.parameters()).device

    # 生成均匀分布的时间步
    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times
    # 生成不均匀分布的时间步
    def get_sampling_timesteps_uneven(self, batch, device, steps):

        # ## samp2:
        # steps1 = 10 
        # steps2 = 7  
        # steps3 = 5   
        # # 生成从1.0到0.8的张量，间隔为0.02
        # times1 = torch.linspace(1.0, 0.8, steps1 + 1, device=device)
        # # 生成从0.8到0.1的张量，间隔为0.1
        # times2 = torch.linspace(0.8, 0.1, steps2 + 1, device=device)
        # # 生成从0.1到0的张量，间隔为0.02
        # times3 = torch.linspace(0.1, 0.0, steps3 + 1, device=device)
        # # 将两个张量连接起来
        # times = torch.cat((times1[:-1], times2))
        # times = torch.cat((times[:-1], times3))
        # times = repeat(times, 't -> b t', b=batch)
        # times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        # times = times.unbind(dim=-1)
        

        ## samp3:
        steps1 = 10 
        steps2 = 8   
        # 生成从1.0到0.8的张量，间隔为0.02
        times1 = torch.linspace(1.0, 0.8, steps1 + 1, device=device)
        # 生成从0.8到0.1的张量，间隔为0.1
        times2 = torch.linspace(0.8, 0.0, steps2 + 1, device=device)
        # 将两个张量连接起来
        times = torch.cat((times1[:-1], times2))
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)

        ## samp4:
        # steps1 = 15 
        # steps2 = 7   
        # # 生成从1.0到0.7的张量，间隔为0.02
        # times1 = torch.linspace(1.0, 0.7, steps1 + 1, device=device)
        # # 生成从0.7到0.1的张量，间隔为0.1
        # times2 = torch.linspace(0.7, 0.0, steps2 + 1, device=device)
        # # 将两个张量连接起来
        # times = torch.cat((times1[:-1], times2))
        # times = repeat(times, 't -> b t', b=batch)
        # times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        # times = times.unbind(dim=-1)
        return times
    
    @torch.no_grad()
    def compute_lambda(self, step, max_steps, start_value=0.0, end_value=1.0):
        return start_value + (end_value - start_value) * (step / max_steps)

    def training_loss(self, img, caption, prop, step, max_steps, *args, **kwargs):
        batch = img.shape[0]
        
        times = torch.zeros(
            (batch,), device=self.device).float().uniform_(0, 1)
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * img + sigma * noise
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.denoise_fn(
                    noised_img, noise_level, caption, prop).detach_()
   
        # pred_uncond = self.denoise_fn(noised_img, noise_level, None, tensor_feature, self_cond)
        pred = self.denoise_fn(noised_img, noise_level, caption, prop, self_cond)

        # w_max = 2.0
        # w = torch.rand(batch, device=self.device) * w_max
        # apply_cond = 1
        # if random() < 0.05:
        #     apply_cond = 0
        # w = w * apply_cond
        # w = w.view(batch, *([1] * (img.ndim - 1)))

        # pred = pred_uncond + w * (pred_cond - pred_uncond)

        # 计算MSE损失
        noise_loss = F.mse_loss(pred, img)
        pred_bin = (pred>0).float()
        img_bin = (img > 0).float() 
        # dice_loss = Dice_loss(pred, img)
       
        # # total_loss = noise_loss+ 0.2 * dice_loss + 0.5 * perceptual_loss 
        # # 计算感知损失
        # perceptual_loss = self.perceptual_loss_fn(pred, img)
        # total_loss = noise_loss + 0.5 * perceptual_loss 
        bce_loss = self.bce_loss_fn(pred_bin, img_bin)

        # (3) 动态融合比例
        lambda_bce = self.compute_lambda(step, max_steps, start_value=0.0, end_value=1.0)

        # (4) 总loss = MSE + 动态加权BCE
        # total_loss = noise_loss + 0.3 * bce_loss
        return noise_loss
    
    @torch.no_grad()
    def sample_with_text(self, caption, prop, batch_size=16,
                           steps=50,  truncated_index:float=0.0, tensor_w: float = 1.0, verbose: bool = True):
        image_size = self.image_size
        shape = (batch_size, 1, image_size, image_size, image_size)
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device=device, steps=steps)
        tensor_zero = - np.ones((4,), dtype=np.float32)  
        tensor_zero = torch.from_numpy(tensor_zero).to(
            device).unsqueeze(0).repeat(batch, 1).to(torch.float32)
        if isinstance(caption, str):
            caption = [caption] * batch
        # assert len(caption) == batch, "Number of caption must match batch size"
        if isinstance(prop, str):
            prop = [prop] * batch
        caption_zero = None
        img = torch.randn(shape, device=device)
        x_start = None

        if verbose:
            _iter = tqdm(time_pairs, desc='sampling loop time step')
        else:
            _iter = time_pairs
        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)
            x_zero_none = self.denoise_fn(
                img, noise_cond, caption_zero, prop, x_start)

            x_cond = self.denoise_fn(img, noise_cond, caption, prop, x_start)
           
            x_start = x_zero_none + tensor_w * (x_cond - x_zero_none)
            # x_start = x_zero_none + tensor_w * \
            #             (self.denoise_fn(img, noise_cond, caption, tensor_zero, x_start) - x_zero_none)

            if time[0] < TRUNCATED_TIME:
                x_start.sign_()

            # DDIM:
            x_start.clamp_(-1, 1)
            pred_noise = (img - alpha * x_start) / \
                            sigma.clamp(min=1e-8)

            img = x_start * alpha_next + pred_noise * sigma_next
        return img

