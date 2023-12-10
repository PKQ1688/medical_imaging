#!/usr/bin/env python
import os

import nibabel as nib
# import matplotlib.pyplot as plt
import numpy as np
import pydicom
from mayavi import mlab

pydicom.config.image_handlers = ["gdcm", "pylibjpeg"]


def load_scan(directory):
    # 加载所有DICOM文件
    files = [
        pydicom.dcmread(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f.endswith(".dcm")
    ]
    # 按照Instance Number排序
    files.sort(key=lambda x: int(x.InstanceNumber))
    # 从DICOM文件中提取图像数据
    return np.stack([s.pixel_array for s in files])


def load_nii_to_numpy(file_path):
    # 加载nii.gz文件
    nii_data = nib.load(file_path)

    # 转换为numpy数组
    image_data = nii_data.get_fdata()

    return np.array(image_data)


def plot_3d(image_3d):
    # 创建一个新的场景
    mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))

    # 添加体数据源
    src = mlab.pipeline.scalar_field(image_3d)

    # 使用体渲染的方式可视化数据
    mlab.pipeline.volume(src, vmin=0, vmax=np.max(image_3d))

    # 启动可视化界面
    mlab.show()


if __name__ == "__main__":
    image = load_scan("data/胃癌化疗150/原图150/1V")
    # image = load_nii_to_numpy("data/monai_spleen/Task09_Spleen/imagesTr/._spleen_2.nii.gz")
    print(image.shape)
    plot_3d(image)
