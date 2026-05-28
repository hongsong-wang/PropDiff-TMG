import fire
import os

from network.model_trainer import DiffusionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from utils.utils import exists
from pytorch_lightning import loggers as pl_loggers
from utils.utils import ensure_directory, run, get_tensorboard_dir, find_best_epoch
from torch.utils.tensorboard import SummaryWriter
import torch
import pdb

torch.set_num_threads(2)

def train_from_folder(
    dataset_folder: str = "/home/daibingxuan/workspace/microstructure_generation_3d/data/datasets",
    results_folder: str = './results',
    voxel_folder: str = "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress",
    csv_path: str = "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_npr.csv",
    name: str = "debug",
    image_size: int = 64,
    base_channels: int = 64,
    optimizier: str = "adam",
    attention_resolutions: str = "4, 8",
    lr: float = 2e-4,
    batch_size: int = 4,
    with_attention: bool = True,
    num_heads: int = 4,
    dropout: float = 0.1,
    noise_schedule: str = "linear",    # 噪声调度策略
    ema_rate: float = 0.999,    # 指数滑动平均的速率
    save_last: bool = True,     # 保存最后一个模型
    verbose: bool = False,     # 是否打印详细信息
    training_epoch: int = 200,
    in_azure: bool = False,
    new: bool = True,
    continue_training: bool = False,
    debug: bool = False,
    use_text_condition: bool = False,
    use_tensor_condition: bool = False,  
    seed: int = 777,
    save_every_epoch: int = 20,
    gradient_clip_val: float = 1.
):  
    if not in_azure:
        debug = True
    else:
        debug = False
    # 创建文件夹
    results_folder = results_folder + "/" + name
    ensure_directory(results_folder)
    if continue_training:
        new = False
    # 删除旧的结果文件
    if new:
        run(f"rm -rf {results_folder}/*")

    # 超参数字典
    model_args = dict(
        results_folder=results_folder,
        dataset_folder=dataset_folder,
        voxel_folder=voxel_folder,
        csv_path=csv_path,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        noise_schedule=noise_schedule,
        use_text_condition=use_text_condition,
        use_tensor_condition=use_tensor_condition,
        base_channels=base_channels,
        optimizier=optimizier,
        attention_resolutions=attention_resolutions,
        with_attention=with_attention,
        num_heads=num_heads,
        dropout=dropout,
        ema_rate=ema_rate,
        verbose=verbose,
        save_every_epoch=save_every_epoch,
        training_epoch=training_epoch,
        gradient_clip_val=gradient_clip_val,
        debug=debug,
    )
    seed_everything(seed)

    model = DiffusionModel(**model_args)
   
  
    if in_azure:
        try:
            log_dir = get_tensorboard_dir()
        except Exception as e:
            log_dir = results_folder
    else:
        log_dir = results_folder

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=None,
        name='logs',
        default_hp_metric=False
    )

    # 保存模型的检查点
    checkpoint_callback = ModelCheckpoint(
        monitor="current_epoch",
        dirpath=results_folder,
        filename="{epoch:02d}",
        save_top_k=10,
        save_last=save_last,
        every_n_epochs=save_every_epoch,
        mode="max",
    )
    best_loss_checkpoint = ModelCheckpoint(
        monitor="loss",             
        dirpath=results_folder,
        filename="best-loss-{epoch:02d}-{loss:.4f}_1",
        save_top_k=1,
        mode="min"
    )
    # 获取最后一个epoch的检查点
    last_epoch = find_best_epoch(results_folder)
    if os.path.exists(os.path.join(results_folder, "last.ckpt")):
        last_ckpt = "last.ckpt"
    else:
        if exists(last_epoch):
            last_ckpt = f"epoch={last_epoch:02d}.ckpt"
        else:
            last_ckpt = "last.ckpt"
    # 初始化训练
    find_unused_parameters = False
    if in_azure:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy=DDPStrategy(find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,
                          max_epochs=training_epoch,
                          log_every_n_steps=10,
                          callbacks=[checkpoint_callback,
                                     best_loss_checkpoint])
    else:
        trainer = Trainer(devices=-1,
                          accelerator="gpu",
                          strategy = DDPStrategy(find_unused_parameters=find_unused_parameters),
                          logger=tb_logger,

                          max_epochs=training_epoch,
                          log_every_n_steps=1,
                          callbacks=[checkpoint_callback,
                                     best_loss_checkpoint])

    
    last_ckpt_path = os.path.join(results_folder, last_ckpt)
    backup_ckpt_path = os.path.join(results_folder, "last_backup.ckpt")
    if continue_training and os.path.exists(last_ckpt_path) and not os.path.exists(backup_ckpt_path):
        import shutil
        shutil.copyfile(last_ckpt_path, backup_ckpt_path)
        print(f"Backup of last checkpoint saved to {backup_ckpt_path}")

    last_ckpt = "best-loss-epoch=1869-loss=0.1245.ckpt"
    if continue_training and os.path.exists(os.path.join(results_folder, last_ckpt)):
        trainer.fit(model, ckpt_path=os.path.join(results_folder, last_ckpt))
    else:
        trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(train_from_folder)
