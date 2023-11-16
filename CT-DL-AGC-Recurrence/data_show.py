#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2023/5/20 22:31
# @Author       : ztn8btc
# @Email        : zhutn@8btc.com
# @LastEditTime : 2023/5/20 22:31
# @File         : data_show.py
import SimpleITK as sitk
# import numpy as np
# import cv2
# import os

path = "data/0517/原序列/1V/"
#
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(path)
reader.SetFileNames(dicom_names)
image = reader.Execute()
image_array = sitk.GetArrayFromImage(image)     # (z,y,x): z:切片数量,y:切片宽,x:切片高

print(image_array.shape)
print(image_array)
# label_file = "data/0517/勾画/1V_Merge.nii.gz"


# def read_img(path):
#     img = sitk.ReadImage(path)
#     data = sitk.GetArrayFromImage(img)
#     print(data.shape)
#     print(data)
#     return data
#
#
# read_img(label_file)
