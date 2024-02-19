#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/18 17:39
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/18 17:39
# @File         : data_handle.py
# from pydicom.pixel_data_handlers.util import apply_voi_lut
import dicom2jpg
import nibabel as nib
import numpy as np
# import pydicom
from PIL import Image


def dicom_to_png(dicom_path, output_path):
    dicom2jpg.dicom2png(dicom_path, output_path)


def nii_to_png(nii_path, png_path, slice_index=-1):
    # 读取NIfTI文件
    nii_image = nib.load(nii_path)
    image_data = nii_image.get_fdata()
    # 如果没有指定切片，选择中间的切片
    if slice_index == -1:
        slice_index = image_data.shape[2] // 2
    # 选择切片并转换为8位格式
    slice_image = Image.fromarray(np.uint8(image_data[:, :, slice_index]))
    # 保存为PNG
    slice_image.save(png_path)


# 使用示例
dicom_to_png("data/862CT/862-data/1V/5-1.dcm", "output_image.png")
nii_to_png("data/862CT/862roi/1V_Merge.nii.gz", "output_image1.png", 0)
