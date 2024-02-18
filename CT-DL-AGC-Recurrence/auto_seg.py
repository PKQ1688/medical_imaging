import os
import shutil

roi_data = "data/Task100/roi"

ori_dcom = "data/ori_dcm"
dcom_list = os.listdir(ori_dcom)


# print(dcom_list)

def copy_folder(src, dst):
    """
    复制src目录到dst目录
    :param src: 源目录
    :param dst: 目标目录
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


for file in os.listdir(roi_data):
    file_dcom = file.split("_")[0]
    # prinfilet(file)
    if file_dcom in dcom_list:
        copy_folder(src=os.path.join(ori_dcom, file_dcom),
                    dst=os.path.join("data/Task100/ori_dcm", file_dcom))
    else:
        print(file_dcom)
