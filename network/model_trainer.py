import copy
from utils.utils import set_requires_grad
from torch.utils.data import DataLoader, ConcatDataset
from network.model_utils import EMA
from network.data_loader_text import VoxelDataset, VoxelDataset1
from pathlib import Path
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import update_moving_average
from pytorch_lightning import LightningModule
from network.model import Propdiff
import torch
import torch.nn as nn
import os
import random

class DiffusionModel(LightningModule):
    def __init__(
        self,
        dataset_folder: str = "",
        results_folder: str = './results',
        voxel_folder: str = "",
        csv_path: str = "",
        image_size: int = 32,
        base_channels: int = 32,
        lr: float = 2e-4,
        batch_size: int = 8,
        attention_resolutions: str = "16,8",
        optimizier: str = "adam",
        with_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        ema_rate: float = 0.999,
        verbose: bool = False,
        save_every_epoch: int = 1,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_text_condition: bool = False,
        use_tensor_condition: bool = False,
        noise_schedule: str = "linear",
        debug: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.results_folder = Path(results_folder)
        self.model = Propdiff(image_size=image_size, base_channels=base_channels,
                                        attention_resolutions=attention_resolutions,
                                        with_attention=with_attention,
                                        dropout=dropout,
                                        use_text_condition=use_text_condition,
                                        use_tensor_condition=use_tensor_condition,
                                        num_heads=num_heads,
                                        noise_schedule=noise_schedule,
                                        verbose=verbose)

        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.dataset_folder = dataset_folder
        self.voxel_folder = voxel_folder
        self.csv_path = csv_path
        self.with_attention = with_attention
        self.save_every_epoch = save_every_epoch
        self.traning_epoch = training_epoch
        self.gradient_clip_val = gradient_clip_val
        self.use_text_condition = use_text_condition,
        self.use_tensor_condition = use_tensor_condition
        
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.optimizier = optimizier
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)
        if debug:
            self.num_workers = 0
        else:
            self.num_workers = 2

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    def configure_optimizers(self):
        if self.optimizier == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizier == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        # **定义 ReduceLROnPlateau 学习率调度器**
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)

        # **返回优化器和调度器**
        return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler
         }

    def train_dataloader(self):
        # _dataset = VoxelDataset(dataset_folder=self.dataset_folder,
        #                         transform=None,
        #                         use_tensor_condition=self.use_tensor_condition
        #                         )
        _dataset = VoxelDataset1(csv_path=self.csv_path,
                                voxel_folder=self.voxel_folder,
                                use_tensor_condition=self.use_tensor_condition
                                )
        
        # _newdataset = NewVoxelDataset(dataset_folder="/home/daibingxuan/workspace/microstructure_generation_3d/active_selected_data",
        #                         transform=None,
        #                         use_tensor_condition=self.use_tensor_condition)
        # _dataset = ConcatDataset([_olddataset, _newdataset])
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=False)
        self.iterations = len(dataloader)
        return dataloader

    def training_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        caption = batch["caption"]
        prop = None
        # if self.use_tensor_condition:
        #     tensor_feature = batch["tensor_feature"]
        # else:
        #     tensor_feature = None
        
        loss = self.model.training_loss(
            occupancy, caption, prop, self.global_step, self.traning_epoch * self.iterations).mean()

        self.log("loss", loss.clone().detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)  
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_val)
        opt.step()


        self.update_EMA()
        # # **添加学习率调度器**
        # scheduler = self.lr_schedulers()
        # scheduler.step(loss)

    def on_train_epoch_end(self):
      
        self.log("current_epoch",self.current_epoch, logger=True)
        return super().on_train_epoch_end()

