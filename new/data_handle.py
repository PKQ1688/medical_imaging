#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/18 17:39
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/18 17:39
# @File         : data_handle.py
import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


def dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    if 'PixelData' in ds:
        # 使用GDCM处理JPEG扩展压缩的DICOM文件
        from pydicom.pixel_data_handlers import gdcm_handler as handler
        if handler.is_available():
            try:
                pixel_array = ds.pixel_array  # 使用GDCM读取像素数组
                # 应用VOI LUT（如果存在）
                pixel_array = apply_voi_lut(pixel_array, ds)
                # 转换为PIL图像并保存
                from PIL import Image
                image = Image.fromarray(pixel_array)
                image.save(output_path)
                print(f"Image saved to {output_path}")
            except Exception as e:
                print(f"Error converting DICOM to PNG: {e}")
        else:
            print("GDCM is not available. Please install it to process this DICOM file.")
    else:
        print("No PixelData found in DICOM file.")


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
