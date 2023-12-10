import os
import pydicom
import numpy as np
import nibabel as nib

def convert_dicom_to_nifti(dicom_folder, output_file):
    # 读取 DICOM 文件
    files = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    
    # 按照切片位置排序文件
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 从 DICOM 文件中提取扫描数据
    image_data = np.stack([file.pixel_array for file in files], axis=-1)

    # 转换数据类型
    image_data = image_data.astype(np.int16)

    # 获取 DICOM 中的仿射矩阵信息
    affine = np.eye(4)
    affine[0,0] = files[0].PixelSpacing[0]
    affine[1,1] = files[0].PixelSpacing[1]
    affine[2,2] = files[0].SliceThickness

    # 创建 NIfTI 图像
    nifti_image = nib.Nifti1Image(image_data, affine)

    # 保存为 .nii.gz
    nib.save(nifti_image, output_file)

# 使用示例
# convert_dicom_to_nifti('data1207/origin_data/00189740V', 'data1207/origin_data/00189740V.nii.gz')

def process_all_dicom_folders(base_folder):
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            output_file = os.path.join(base_folder, f"{folder_name}.nii.gz")
            print(f"Converting {folder_path} to {output_file}")
            try:
                convert_dicom_to_nifti(folder_path, output_file)
            except Exception as e:
                print(f"Failed to convert {folder_path}: {e}")

process_all_dicom_folders('data1207/origin_data')
