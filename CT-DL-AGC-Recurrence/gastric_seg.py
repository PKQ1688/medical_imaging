#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2023/12/13 23:24
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2023/12/13 23:24
# @File         : gastric_seg.py
import glob
import os

from monai.data import DataLoader, Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import first, set_determinism
import matplotlib.pyplot as plt


def load_data(data_dir):
    """
    Loads data from a local file
    """
    # 读取本地文件
    train_images = sorted(glob.glob(os.path.join(data_dir, "origin_data", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "roi_data", "*.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    # print(len(data_dicts))
    train_files, val_files = data_dicts[:-20], data_dicts[-20:]

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
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
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
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    return train_transforms, val_transforms


def main():
    set_determinism(seed=0)

    data_dir = "data/data1207"
    train_files, val_files = load_data(data_dir)
    train_transforms, val_transforms = data_transforms()
    print(val_files)

    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    print(len(check_data["image"]))
    print(len(check_data["label"]))
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 120], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 120])
    plt.show()


if __name__ == '__main__':
    main()
