#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/1/10 11:11
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/1/10 11:11
# @File         : spleen_segmentation_3d_visualization.py
import glob
import os

import aim
import torch
from aim.pytorch import track_gradients_dists, track_params_dists
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
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
import monai

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# data_dir = "data/Task09_Spleen"
# # data_dir = "data/temp_data"

# train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
# train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

# data_dicts = [
#     {"image": image_name, "label": label_name}
#     for image_name, label_name in zip(train_images, train_labels)
# ]
# train_files, val_files = data_dicts[:20], data_dicts[:20]

data_dir = "data/Task400"
train_images = sorted(glob.glob(os.path.join(data_dir, "ori_data", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "ROI400", "*.nii.gz")))

# question_img_id = ["00200100", "00205095", "00206507", "00163639"]
# question_img_list = [f"data/ori_data/{id}_Merge.nii.gz" for id in question_img_id]

data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
    # if image_name not in question_img_list
]
print(len(data_dicts))
# exit()

# train_files, val_files = data_dicts[:2000], data_dicts[2000:]
train_files, val_files = data_dicts[:320], data_dicts[320:]

print(f"training samples: {len(train_files)}, validation samples: {len(val_files)}")

set_determinism(seed=0)

# 0.5是放大图片,2.0是缩小。 只定向于第三位
spacing = (1.5, 1.5, 0.5)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        # Resize(keys=["image", "label"] ,spatial_size=(512, 512, 128)),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
    ]
)

train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=0.3, num_workers=0
)

# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)

# val_ds = CacheDataset(
    # data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0
# )
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
# device = torch.device("cuda:0")

UNet_meatdata = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 2,
    "channels": (16, 32, 64, 128, 256),
    "strides": (2, 2, 2, 2),
    "num_res_units": 2,
    "norm": Norm.BATCH,
}

model = UNet(**UNet_meatdata).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
# loss_function = DiceLoss()
loss_type = "DiceLoss"
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

Optimizer_metadata = {}
for ind, param_group in enumerate(optimizer.param_groups):
    optim_meta_keys = list(param_group.keys())
    Optimizer_metadata[f"param_group_{ind}"] = {
        key: value for (key, value) in param_group.items() if "params" not in key
    }

max_epochs = 400
val_interval = 10
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# initialize a new Aim Run
aim_run = aim.Run(experiment="2张图片进行测试")
# log model metadata
aim_run["UNet_meatdata"] = UNet_meatdata
# log optimizer metadata
aim_run["Optimizer_metadata"] = Optimizer_metadata

slice_to_track = 80

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
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
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
                roi_size = (160, 160, 160)
                # roi_size = (96, 96, 16)
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
                    model.state_dict(), os.path.join("model", "best_metric_model.pth")
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

# finalize Aim Run
aim_run.close()

print(
    f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}"
)
