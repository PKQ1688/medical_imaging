import os

import SimpleITK as sitk


# 读取示例 nii.gz 文件
def read_nii(file_path):
    nii_image = sitk.ReadImage(file_path)
    affine = nii_image.GetDirection()
    voxel_spacing = nii_image.GetSpacing()
    return affine, voxel_spacing


# 转换 DICOM 到 NIfTI
def convert_dicom_to_nifti(dicom_directory, output_file, affine, voxel_spacing):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    image.SetDirection(affine)
    image.SetSpacing(voxel_spacing)

    sitk.WriteImage(image, output_file, True)  # True for compressing to .nii.gz


# 示例文件路径
# example_nii_path = 'data/data1207/roi_data/00189740_Merge.nii.gz'  # 替换为您的示例 NIfTI 文件路径
# dicom_directory = 'data/data1207/origin_dicom/00189740V'  # 替换为您的 DICOM 文件夹路径
# output_directory = '00189740.nii.gz'  # 指定输出目录


def handle_all_data(origin_dicom_path, roi_data_path, output_data_path):
    # 读取示例 nii.gz 文件
    dicom_list = sorted(os.listdir(origin_dicom_path))
    roi_list = sorted(os.listdir(roi_data_path))

    for dicom_directory, example_nii_path in zip(dicom_list, roi_list):
        # print(dicom_directory)
        # print(example_nii_path)

        # 读取示例文件的属性
        example_affine, example_voxel_spacing = read_nii(
            os.path.join(roi_data_path, example_nii_path)
        )

        output_directory = os.path.join(output_data_path, example_nii_path)
        # # 转换并调整新 NIfTI 文件以匹配示例文件
        convert_dicom_to_nifti(
            os.path.join(origin_dicom_path, dicom_directory),
            output_directory,
            example_affine,
            example_voxel_spacing,
        )
        # break


if __name__ == "__main__":
    handle_all_data(
        origin_dicom_path="data/data1207/origin_dicom",
        roi_data_path="data/data1207/roi_data",
        output_data_path="data/data1207/origin_data",
    )
