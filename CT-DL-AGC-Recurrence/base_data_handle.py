import os

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pydicom


def save_dicon_data(dicom_dir, save_path):
    # 读取DICOM图像
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    # 将图像保存为NIfTI格式
    sitk.WriteImage(image, save_path)


def load_dicom_series(directory):
    """ 读取DICOM序列并重新定向为RAI方向 """
    files = [pydicom.dcmread(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith('.dcm')]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # 从 DICOM 文件中提取扫描数据
    # return np.stack([file.pixel_array for file in files],axis=-1)
    image_stack = np.stack([file.pixel_array for file in files], axis=-1)
    # RAS到RAI的转换，仅需要沿着Z轴翻转
    # image_stack = np.flip(image_stack, axis=0)
    image_stack = np.flip(image_stack, axis=1)
    # image_stack = np.flip(image_stack, axis=2)
    image_stack_rotated = np.rot90(image_stack, k=1, axes=(0, 1))
    return image_stack_rotated


def convert_dicom_to_nifti(dicom_series, output_path):
    """ 将DICOM序列转换为NIfTI格式 """
    data = dicom_series.astype(np.int16)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, output_path)


def handel_dir_dicom(dicom_dir, nifti_path):
    for file_name in os.listdir(dicom_dir):
        dicom_path_dir = os.path.join(dicom_dir, file_name)
        # dicom_series = load_dicom_series(dicom_path_dir)
        nifti_output_path = os.path.join(nifti_path, file_name + '.nii.gz')
        # convert_dicom_to_nifti(dicom_series, nifti_output_path)
        save_dicon_data(dicom_path_dir, nifti_output_path)


def convert_rai_to_lpi(input_file_path, output_file_path):
    for file_name in os.listdir(input_file_path):
        input_file = os.path.join(input_file_path, file_name)
        img = nib.load(input_file)
        data = img.get_fdata()

        # No change needed for I to I
        converted_data = np.flip(data, (0, 1))
        new_affine = img.affine.copy()
        new_affine[0, :] *= -1  # Flip X-axis
        new_affine[1, :] *= -1  # Flip Y-axi

        # Create a new Nifti1Image with the converted data
        converted_img = nib.Nifti1Image(converted_data, new_affine)

        # Save the converted image
        nib.save(converted_img, os.path.join(output_file_path, file_name))


if __name__ == '__main__':
    _dicom_dir = '/Users/zhutaonan/Downloads/data1207/复发验证集-V120/'
    _nifti_path = 'data/data1207/origin_data'
    handel_dir_dicom(_dicom_dir, _nifti_path)
    # convert_rai_to_lpi("data/data1207/roi_data", "data/data1207/roi_ras")
