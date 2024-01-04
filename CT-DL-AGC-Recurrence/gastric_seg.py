#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2023/12/13 23:24
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2023/12/13 23:24
# @File         : gastric_seg.py
import glob
import os

import aim
import torch
from aim.pytorch import track_gradients_dists, track_params_dists
from loguru import logger
from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    # IterableDataset,
    Dataset,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

# from monai.networks.layers import Norm
from monai.networks.nets import UNETR
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

from torch.cuda.amp import GradScaler, autocast

# device = torch.device("cuda:0")
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir):
    """
    Loads data from a local file
    """
    # 读取本地文件
    train_images = sorted(glob.glob(os.path.join(data_dir, "ori_data", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "roi", "*.nii.gz")))

    # train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    # train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    # print(len(data_dicts))
    # train_files, val_files = data_dicts[:-200], data_dicts[-200:]
    train_files, val_files = data_dicts[:200], data_dicts[200:300]

    return train_files, val_files


def data_transforms():
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"], source_key="image", allow_smaller=True
            ),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 0.5),
                mode=("bilinear", "nearest"),
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                # spatial_size=(96, 96, 96),
                spatial_size=(96, 96, 16),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"], source_key="image", allow_smaller=True
            ),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )
    return train_transforms, val_transforms


# def data_transforms():
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(1.5, 1.5, 2.0),
#                 mode=("bilinear", "nearest"),
#             ),
#             ScaleIntensityRanged(
#                 keys=["image"],
#                 a_min=-175,
#                 a_max=250,
#                 b_min=0.0,
#                 b_max=1.0,
#                 clip=True,
#             ),
#             CropForegroundd(keys=["image", "label"], source_key="image"),
#             RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=(96, 96, 96),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="image",
#                 image_threshold=0,
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[0],
#                 prob=0.10,
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[1],
#                 prob=0.10,
#             ),
#             RandFlipd(
#                 keys=["image", "label"],
#                 spatial_axis=[2],
#                 prob=0.10,
#             ),
#             RandRotate90d(
#                 keys=["image", "label"],
#                 prob=0.10,
#                 max_k=3,
#             ),
#             RandShiftIntensityd(
#                 keys=["image"],
#                 offsets=0.10,
#                 prob=0.50,
#             ),
#         ]
#     )
#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(1.5, 1.5, 2.0),
#                 mode=("bilinear", "nearest"),
#             ),
#             ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
#             CropForegroundd(keys=["image", "label"], source_key="image"),
#         ]
#     )
#     return train_transforms, val_transforms


def get_model():
    # UNet_metadata = {
    #     "spatial_dims": 3,
    #     "in_channels": 1,
    #     "out_channels": 2,
    #     "channels": (16, 32, 64, 128, 256),
    #     "strides": (2, 2, 2, 2),
    #     "num_res_units": 2,
    #     "norm": Norm.BATCH,
    # }

    # model = UNet(**UNet_metadata).to(device)

    UNETR_metadata = {
        "in_channels": 1,
        "out_channels": 2,
        "img_size": (96, 96, 16),
        "feature_size": 16,
        "hidden_size": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        # "pos_embed": "perceptron",
        "proj_type": "conv",
        "norm_name": "instance",
        "res_block": True,
        "conv_block": True,
        "dropout_rate": 0.0,
    }

    model = UNETR(**UNETR_metadata).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    loss_type = "DiceLoss"
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    Optimizer_metadata = {}
    for ind, param_group in enumerate(optimizer.param_groups):
        # optim_meta_keys = list(param_group.keys())
        Optimizer_metadata[f"param_group_{ind}"] = {
            key: value for (key, value) in param_group.items() if "params" not in key
        }

    return (
        # UNet_metadata,
        UNETR_metadata,
        Optimizer_metadata,
        model,
        optimizer,
        loss_function,
        loss_type,
        dice_metric,
    )


def train(train_loader, val_loader, train_ds, val_ds, aim_run):
    max_epochs = 600
    val_interval = 10
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    (
        UNETR_metadata,
        Optimizer_metadata,
        model,
        optimizer,
        loss_function,
        loss_type,
        dice_metric,
    ) = get_model()

    # log model metadata
    aim_run["UNETR_metadata"] = UNETR_metadata
    # log optimizer metadata
    aim_run["Optimizer_metadata"] = Optimizer_metadata

    slice_to_track = 50

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            # loss.backward()
            # optimizer.step()

            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            # 调整比例和更新权重
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}"
            )
            # track batch loss metric
            aim_run.track(loss.item(), name="batch_loss", context={"type": loss_type})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        # track epoch loss metric
        aim_run.track(epoch_loss, name="epoch_loss", context={"type": loss_type})

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            if (epoch + 1) % val_interval * 2 == 0:
                # track model params and gradients
                track_params_dists(model, aim_run)
                # THIS SEGMENT TAKES RELATIVELY LONG (Advise Against it)
                track_gradients_dists(model, aim_run)

            model.eval()
            with torch.no_grad():
                for index, val_data in enumerate(val_loader):
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    # roi_size = (160, 160, 160)
                    roi_size = (96, 96, 16)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )

                    # tracking input, label and output images with Aim
                    output = torch.argmax(val_outputs, dim=1)[
                        0, :, :, slice_to_track
                    ].float()

                    aim_run.track(
                        aim.Image(
                            val_inputs[0, 0, :, :, slice_to_track],
                            caption=f"Input Image: {index}",
                        ),
                        name="validation",
                        context={"type": "input"},
                    )
                    aim_run.track(
                        aim.Image(
                            val_labels[0, 0, :, :, slice_to_track],
                            caption=f"Label Image: {index}",
                        ),
                        name="validation",
                        context={"type": "label"},
                    )
                    aim_run.track(
                        aim.Image(output, caption=f"Predicted Label: {index}"),
                        name="predictions",
                        context={"type": "labels"},
                    )

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # track val metric
                aim_run.track(metric, name="val_metric", context={"type": loss_type})

                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join("model", "best_metric_model.pth"),
                    )

                    best_model_log_message = (
                        f"saved new best metric model at the {epoch + 1}th epoch"
                    )
                    aim_run.track(
                        aim.Text(best_model_log_message),
                        name="best_model_log_message",
                        epoch=epoch + 1,
                    )
                    print(best_model_log_message)

                message1 = f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                message2 = f"\nbest mean dice: {best_metric:.4f} "
                message3 = f"at epoch: {best_metric_epoch}"

                aim_run.track(
                    aim.Text(message1 + "\n" + message2 + message3),
                    name="epoch_summary",
                    epoch=epoch + 1,
                )
                print(message1, message2, message3)

    return best_metric, best_metric_epoch


def run_pipeline():
    set_determinism(seed=0)
    # data_dir = "data/data1207"
    data_dir = "data/"
    # data_dir = "data/Task09_Spleen"
    train_files, val_files = load_data(data_dir)
    train_transforms, val_transforms = data_transforms()
    logger.info(train_files)

    from monai.data.utils import pad_list_data_collate

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0
    )
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
    )

    # val_ds = CacheDataset(
    #     data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=0
    # )
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
    )

    # initialize a new Aim Run
    aim_run = aim.Run()
    best_metric, best_metric_epoch = train(
        train_loader, val_loader, train_ds, val_ds, aim_run
    )
    aim_run.close()
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )


if __name__ == "__main__":
    run_pipeline()
