#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/18 17:39
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/18 17:39
# @File         : data_handle.py
# from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
import pydicom
import nibabel as nib
import numpy as np

# import pydicom
import cv2
from PIL import Image
from tqdm import tqdm


def lin_stretch_img(img, low_prc, high_prc, do_ignore_minmax=True):
    """
    Apply linear "stretch" - low_prc percentile goes to 0,
    and high_prc percentile goes to 255.
    The result is clipped to [0, 255] and converted to np.uint8

    Additional feature:
    When computing high and low percentiles, ignore the minimum and maximum intensities (assumed to be outliers).
    """
    # For ignoring the outliers, replace them with the median value
    if do_ignore_minmax:
        tmp_img = img.copy()
        med = np.median(img)  # Compute median
        tmp_img[img == img.min()] = med
        tmp_img[img == img.max()] = med
    else:
        tmp_img = img

    lo, hi = np.percentile(
        tmp_img, (low_prc, high_prc)
    )  # Example: 1% - Low percentile, 99% - High percentile

    if lo == hi:
        return np.full(
            img.shape, 128, np.uint8
        )  # Protection: return gray image if lo = hi.

    stretch_img = (img.astype(float) - lo) * (
        255 / (hi - lo)
    )  # Linear stretch: lo goes to 0, hi to 255.
    stretch_img = stretch_img.clip(0, 255).astype(
        np.uint8
    )  # Clip range to [0, 255] and convert to uint8
    return stretch_img


def dicom_to_png(dicom_path, output_path):
    dcm_list = [f for f in os.listdir(dicom_path)]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for f in dcm_list:  # remove "[:10]" to convert all images
        ds = pydicom.read_file(dicom_path + f)  # read dicom image
        img = ds.pixel_array  # get image array
        img = lin_stretch_img(img, 1, 99)
        cv2.imwrite(output_path + f.replace(".dcm", ".png"), img)  # write png image


def nii_to_png(nii_path, png_path):
    if not os.path.exists(png_path):
        os.mkdir(png_path)
    # 读取NIfTI文件
    nii_image = nib.load(nii_path)
    image_data = nii_image.get_fdata()
    # 如果没有指定切片，选择中间的切片
    # if slice_index == -1:
    # slice_index = image_data.shape[2] // 2
    # 选择切片并转换为8位格式
    for slice_index in range(image_data.shape[2]):
        slice_image = Image.fromarray(np.uint8(image_data[:, :, slice_index]))
        # slice_image = (slice_image * 255).astype(np.uint8)
        # 保存为PNG
        slice_image.save(f"{png_path}/{slice_index+1}.png")


# 使用示例
# dicom_to_png("data/862CT/862-data/1V/", "1V/")
# nii_to_png("data/862CT/862roi/1V_Merge.nii.gz", "1V_mask/")

dicom_data_path = "data/862CT/862-data/"
dicom_data_path_list = os.listdir(dicom_data_path)

for dicom_name in tqdm(dicom_data_path_list, total=len(dicom_data_path_list)):
    # print(dicom_name)
    dicom_to_png(
        f"data/862CT/862-data/{dicom_name}/",
        f"data/862CT/res_data/origin_img/{dicom_name}/",
    )
    nii_to_png(
        f"data/862CT/862roi/{dicom_name}_Merge.nii.gz",
        f"data/862CT/res_data/mask_img/{dicom_name}/",
    )
    # break
