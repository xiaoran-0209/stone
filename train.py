import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.ore_unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# -------------------------------------#
#   强化版早停回调类
# -------------------------------------#
class EvalCallback:
    def __init__(self, model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
                 eval_flag=True, period=5, patience=10, delta=0.001, save_best=True):
        self.model = model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.VOCdevkit_path = VOCdevkit_path
        self.log_dir = log_dir
        self.Cuda = Cuda
        self.eval_flag = eval_flag
        self.period = period
        self.patience = patience
        self.delta = delta
        self.save_best = save_best

        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def on_epoch_end(self, epoch, val_loss):
        if not self.eval_flag or epoch % self.period != 0:
            return

        # 检查验证损失是否显著下降
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.save_best and torch.distributed.get_rank() == 0:
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, f'best_epoch_weights.pth'))
                print(f'\nBest model saved at epoch {epoch}, val_loss={val_loss:.4f}')
        else:
            self.counter += 1
            print(f'[EarlyStopping] Counter: {self.counter}/{self.patience}, Best Val Loss: {self.best_val_loss:.4f}')
            if self.counter >= self.patience:
                self.early_stop = True


# -------------------------------------#
#   数据增强配置（与UnetDataset兼容）
# -------------------------------------#
def get_train_transform(input_shape):
    return A.Compose([
        A.RandomResizedCrop(
            height=input_shape[0],
            width=input_shape[1],
            scale=(0.08, 1.0)),  # 面积比例范围
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3),
        A.GaussNoise(var_limit=(10, 30)),
        A.Blur(blur_limit=3, p=0.1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform(input_shape):
    return A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# -------------------------------------#
#   修改后的UnetDataset包装器
# -------------------------------------#
class EnhancedUnetDataset(UnetDataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, transform=None):
        super().__init__(annotation_lines, input_shape, num_classes, train, dataset_path)
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        # 调用父类获取数据（假设返回的 jpg 是 CHW 格式的 NumPy 数组）
        jpg, png, seg_labels = super().__getitem__(index)

        if self.transform and self.train:
            # 调整维度顺序：CHW -> HWC
            jpg = jpg.transpose(1, 2, 0)  # 使用 NumPy 的 transpose，而非 PyTorch 的 permute
            # 应用数据增强
            transformed = self.transform(image=jpg, mask=png)
            jpg = transformed['image']  # Albumentations 自动转换为张量（若包含 ToTensorV2）
            png = transformed['mask']

        return jpg, png, seg_labels


if __name__ == "__main__":
    # -------------------------------------#
    #   参数配置（优化版）
    # -------------------------------------#
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = True  # 启用混合精度

    # 模型参数
    num_classes = 21
    backbone = "vgg"
    pretrained = False
    model_path = "model_data/unet_vgg_voc.pth"
    input_shape = [512, 512]

    # 训练策略
    Init_Epoch = 0
    Freeze_Epoch = 30
    Freeze_batch_size = 4
    UnFreeze_Epoch = 80
    Unfreeze_batch_size = 2
    Freeze_Train = True
    UnFreeze_flag = False
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    # 优化器
    optimizer_type = "adamw"
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    weight_decay = 1e-4  # 权重衰减
    momentum = 0.9
    cuda = True
    # 损失函数
    dice_loss = True
    focal_loss = False

    #cls_weights = np.array([1.0, 2.0], np.float32)  # 调整类别权重
    cls_weights     = np.ones([num_classes], np.float32)


    # 回调设置
    eval_flag = True
    eval_period = 3
    save_period = 5
    save_dir = 'logs'
    VOCdevkit_path = 'VOCdevkit'

    # 早停参数
    early_stopping_patience = 10
    early_stopping_delta = 0.001

    # -------------------------------------#
    #   设备初始化
    # -------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    # -------------------------------------#
    #   模型初始化（添加dropout_rate参数）
    # -------------------------------------#
    model = Unet(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone,
        dropout_rate=0.2  # 需在Unet类中添加该参数
    ).train()
    model_train = model.train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # -------------------------------------#
    #   优化器设置
    # -------------------------------------#
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Init_lr,
        betas=(momentum, 0.999),
        weight_decay=weight_decay
    )

    # -------------------------------------#
    #   数据加载（兼容原始UnetDataset）
    # -------------------------------------#
    train_transform = get_train_transform(input_shape)
    val_transform = get_val_transform(input_shape)

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    train_dataset = EnhancedUnetDataset(
        train_lines, input_shape, num_classes, True, VOCdevkit_path, transform=train_transform
    )
    val_dataset = EnhancedUnetDataset(
        val_lines, input_shape, num_classes, False, VOCdevkit_path, transform=val_transform
    )
    epoch_step = len(train_dataset) // Freeze_batch_size
    epoch_step_val = len(val_dataset) // Freeze_batch_size

    gen = DataLoader(
        train_dataset,
        batch_size=Freeze_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=unet_dataset_collate
    )
    gen_val = DataLoader(
        val_dataset,
        batch_size=Freeze_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=unet_dataset_collate
    )

    # -------------------------------------#
    #   回调函数
    # -------------------------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)

        eval_callback = EvalCallback(
            model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda,
            eval_flag=eval_flag,
            period=eval_period,
            patience=early_stopping_patience,
            delta=early_stopping_delta
        )
    else:
        loss_history = None
        eval_callback = None

    # -------------------------------------#
    #   训练主循环
    # -------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 解冻逻辑
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            model.unfreeze_backbone()
            batch_size = Unfreeze_batch_size
            gen = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=unet_dataset_collate
            )
            gen_val = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=unet_dataset_collate
            )
            UnFreeze_flag = True

        # 训练一个epoch（修复 Epoch -> UnFreeze_Epoch）
        val_loss = fit_one_epoch(
            model_train=model,
            model=model,
            loss_history=loss_history,
            eval_callback=eval_callback,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_step_val=epoch_step_val,
            gen=gen,
            gen_val=gen_val,
            Epoch=UnFreeze_Epoch,  # 关键修复点：传递已定义的变量
            cuda=Cuda,
            dice_loss=dice_loss,
            focal_loss=focal_loss,
            cls_weights=cls_weights,
            num_classes=num_classes,
            fp16=fp16,
            scaler=scaler,
            save_period=save_period,
            save_dir=save_dir,
            local_rank=0
        )

        # 早停检查
        if eval_callback and eval_callback.early_stop:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    if local_rank == 0:
        loss_history.writer.close()