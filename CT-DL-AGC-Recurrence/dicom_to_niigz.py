import SimpleITK as sitk
import os

def readdcm(filepath):
    # 创建一个SimpleITK的图像系列阅读器
    reader = sitk.ImageSeriesReader()  
    reader.MetaDataDictionaryArrayUpdateOn()  
    reader.LoadPrivateTagsOn()  
    # 获取DICOM系列的ID
    series_id = reader.GetGDCMSeriesIDs(filepath)
    # 获取DICOM系列中的文件名
    series_file_names = reader.GetGDCMSeriesFileNames(filepath, series_id[0])

    # 设置阅读器的文件名
    reader.SetFileNames(series_file_names)  
    # 执行读取操作，获取图像
    images = reader.Execute()  

    return images

dcm_path = r'F:\test'
save_path = 'datafile/400data/datas/'

file_root = 'datafile/400data/data400/' 

file_list = os.listdir(file_root)
print(file_list)
for img_name in file_list:
    #if img_name.endswith('.dcm'):
    dcm_path = file_root + img_name
    print(dcm_path)

     # 调用readdcm函数读取DICOM图像
    dcm_images = readdcm(dcm_path)
    # 将图像保存为.nii.gz格式
    sitk.WriteImage(dcm_images, os.path.join('{}.nii.gz'.format(dcm_path)))