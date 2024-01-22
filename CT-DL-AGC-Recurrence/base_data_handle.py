import os
import re

import SimpleITK as sitk
from loguru import logger

# import nibabel as nib
# import numpy as np
# import pydicom
from tqdm import tqdm


# 读取示例 nii.gz 文件
def read_nii(file_path):
    # 使用simpleitk读取nii.gz文件
    # logger.info(f"read nii file: {file_path}")
    nii_image = sitk.ReadImage(file_path)
    affine = nii_image.GetDirection()
    voxel_spacing = nii_image.GetSpacing()
    return affine, voxel_spacing

    # # 使用nibabel读取nii.gz文件
    # nii_image = nib.load(file_path)
    # affine = nii_image.affine
    # voxel_spacing = nii_image.header.get_zooms()[:3]
    # return affine, voxel_spacing


# 转换 DICOM 到 NIfTI
def convert_dicom_to_nifti_v1(dicom_directory, output_file, affine, voxel_spacing):
    reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    image.SetDirection(affine)
    image.SetSpacing(voxel_spacing)

    sitk.WriteImage(image, output_file, True)  # True for compressing to .nii.gz


# 转换 DICOM 到 NIfTI
# def convert_dicom_to_nifti_v2(dicom_directory, output_file, affine, voxel_spacing):
#     # 读取所有DICOM文件
#     dicom_files = [
#         os.path.join(dicom_directory, f)
#         for f in os.listdir(dicom_directory)
#         if f.endswith(".dcm")
#     ]
#     dicom_files.sort()  # 确保文件是有序的
#
#     # 读取DICOM文件的像素数据
#     pixel_arrays = [pydicom.dcmread(f).pixel_array for f in dicom_files]
#     volume = np.stack(pixel_arrays, axis=-1)
#
#     # 创建NIfTI图像
#     nii_image = nib.Nifti1Image(volume, affine)
#     nii_image.header.set_zooms(voxel_spacing + (nii_image.header.get_zooms()[3],))
#
#     # 保存NIfTI图像
#     nib.save(nii_image, output_file)


# 示例文件路径
# example_nii_path = 'data/data1207/roi_data/00189740_Merge.nii.gz'  # 替换为您的示例 NIfTI 文件路径
# dicom_directory = 'data/data1207/origin_dicom/00189740V'  # 替换为您的 DICOM 文件夹路径
# output_directory = '00189740.nii.gz'  # 指定输出目录


def handle_all_data(origin_dicom_path, roi_data_path, output_data_path):
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
    # 读取示例 nii.gz 文件
    dicom_list = sorted(os.listdir(origin_dicom_path))
    roi_list = sorted(os.listdir(roi_data_path))
    logger.info(len(roi_list))
    logger.info(len(dicom_list))
    # roi_list = roi_list[:30]

    # dicom_list.remove(".DS_Store")
    # roi_list.remove(".DS_Store")

    # for dicom_directory, example_nii_path in tqdm(
    #         zip(dicom_list, roi_list), total=len(dicom_list)
    # ):
    # for dicom_directory in tqdm(dicom_list,total=len(dicom_list)):
    for example_nii_path in tqdm(roi_list, total=len(roi_list)):
        # example_nii_path = dicom_directory.replace("V", "") + "_Merge.nii.gz"
        dicom_directory_v1 = example_nii_path.replace("_Merge.nii.gz", "") + "V"
        dicom_directory_v2 = example_nii_path.replace("_Merge.nii.gz", "")
        # print(dicom_directory)
        # print(example_nii_path)
        if dicom_directory_v1 in dicom_list:
            dicom_directory = dicom_directory_v1
        elif dicom_directory_v2 in dicom_list:
            dicom_directory = dicom_directory_v2
        else:
            logger.error(f"dicom_directory:{dicom_directory_v1} not in dicom_list")

        pattern = r"\d+"  # 匹配一个或多个数字
        dicom_num = re.findall(pattern, dicom_directory)
        example_num = re.findall(pattern, example_nii_path)

        assert dicom_num == example_num

        # 读取示例文件的属性
        example_affine, example_voxel_spacing = read_nii(
            os.path.join(roi_data_path, example_nii_path)
        )

        output_directory = os.path.join(output_data_path, example_nii_path)
        # # 转换并调整新 NIfTI 文件以匹配示例文件
        logger.info(f"dicom_directory:{dicom_directory}")
        logger.info(f"example_nii_path:{example_nii_path}")
        convert_dicom_to_nifti_v1(
            os.path.join(origin_dicom_path, dicom_directory),
            output_directory,
            example_affine,
            example_voxel_spacing,
        )
        # break


if __name__ == "__main__":
    handle_all_data(
        origin_dicom_path=f"data/Task100/ori_dcm/",
        roi_data_path=f"data/Task100/roi/",
        output_data_path=f"data/Task100/ori_data/",
    )
