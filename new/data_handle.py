#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/2/18 17:39
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/2/18 17:39
# @File         : data_handle.py
# from pydicom.pixel_data_handlers.util import apply_voi_lut
import json
import os

# import pydicom
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm


def lin_stretch_img(img, low_prc, high_prc, do_ignore_minmax=True):
    """
    Apply linear "stretch" - low_prc percentile goes to 0,
    and high_prc percentile goes to 255.
    The result is clipped to [0, 255] and converted to np.uint8

    Additional feature:
    When computing high and low percentiles, ignore the minimum and maximum intensities (assumed to be outliers).
    """
    # For ignoring the outliers, replace them with the median value
    if do_ignore_minmax:
        tmp_img = img.copy()
        med = np.median(img)  # Compute median
        tmp_img[img == img.min()] = med
        tmp_img[img == img.max()] = med
    else:
        tmp_img = img

    lo, hi = np.percentile(
        tmp_img, (low_prc, high_prc)
    )  # Example: 1% - Low percentile, 99% - High percentile

    if lo == hi:
        return np.full(
            img.shape, 128, np.uint8
        )  # Protection: return gray image if lo = hi.

    stretch_img = (img.astype(float) - lo) * (
        255 / (hi - lo)
    )  # Linear stretch: lo goes to 0, hi to 255.
    stretch_img = stretch_img.clip(0, 255).astype(
        np.uint8
    )  # Clip range to [0, 255] and convert to uint8
    return stretch_img


def set_window_level(data, window_width, window_level):
    """Apply window width and level to the image data."""
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    data = np.clip(data, lower_bound, upper_bound)
    return (data - lower_bound) / (upper_bound - lower_bound) * 255


def dicom_to_png(dicom_path, output_path):
    dcm_list = [f for f in os.listdir(dicom_path)]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for f in dcm_list:  # remove "[:10]" to convert all images
        ds = pydicom.dcmread(dicom_path + f)
        pixel_array = ds.pixel_array.astype(np.float32)

        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        windowed_image = set_window_level(pixel_array, 400, 30)

        # Convert to uint8
        image_8bit = windowed_image.astype(np.uint8)

        # Save as PNG
        plt.imsave(output_path + f.replace(".dcm", ".png"), image_8bit, cmap="gray")

        # 将DICOM像素数组转换为适合显示的像素值

        # print("success")
        # img = ds.pixel_array  # get image array
        # img = lin_stretch_img(img, 1, 99)
        # cv2.imwrite(output_path + f.replace(".dcm", ".png"), img)  # write png image


# def nii_to_png(nii_path, png_path):
#     if not os.path.exists(png_path):
#         os.mkdir(png_path)
#     # 读取NIfTI文件
#     nii_image = nib.load(nii_path)
#     image_data = nii_image.get_fdata().astype(np.float32)
#     # 如果没有指定切片，选择中间的切片
#     # if slice_index == -1:
#     # slice_index = image_data.shape[2] // 2
#
#     # Save as PNG
#     for slice_index in range(image_data.shape[2]):
#         slice_data = image_data[:, :, slice_index]
#         windowed_image = set_window_level(slice_data, 400, 30)
#         image_8bit = windowed_image.astype(np.uint8)
#         # image_8bit *= 255
#
#         # Save as PNG
#         plt.imsave(f"{png_path}/{slice_index+1}.png", image_8bit, cmap="gray")

def nii_to_png(nii_path, png_path):
    if not os.path.exists(png_path):
        os.mkdir(png_path)
    # 读取NIfTI文件
    nii_image = nib.load(nii_path)
    image_data = nii_image.get_fdata()
    # 如果没有指定切片，选择中间的切片
    # if slice_index == -1:
    # slice_index = image_data.shape[2] // 2
    # 选择切片并转换为8位格式
    for slice_index in range(image_data.shape[2]):
        slice_image = Image.fromarray(np.uint8(image_data[:, :, slice_index]))
        # slice_image = (slice_image * 255).astype(np.uint8)
        # 保存为PNG
        slice_image.save(f"{png_path}/{slice_index+1}.png")

# 使用示例
# dicom_to_png("data/862CT/862-data/1V/", "1V/")
# nii_to_png("data/862CT/862roi/1V_Merge.nii.gz", "1V_mask/")


def get_png_data():
    dicom_data_path = "data/862CT/862-data/"
    dicom_data_path_list = os.listdir(dicom_data_path)

    for dicom_name in tqdm(dicom_data_path_list, total=len(dicom_data_path_list)):
        # print(dicom_name)
        # dicom_to_png(
        #     f"data/862CT/862-data/{dicom_name}/",
        #     f"data/862CT/res_data/origin_img/{dicom_name}/",
        # )
        nii_to_png(
            f"data/862CT/862roi/{dicom_name}_Merge.nii.gz",
            f"data/862CT/res_data/mask_img/{dicom_name}/",
        )
        # break


def get_use_data():
    mask_data_list = os.listdir("data/862CT/res_data/mask_img/")
    for mask_name in mask_data_list:
        print(mask_name)
        if mask_name in ["80V", "97V"]:
            continue

        mask_img_list_small = sorted(
            os.listdir(f"data/862CT/res_data/mask_img/{mask_name}"),
            key=lambda x: int(x.split(".")[0]),
        )
        img_list_small = sorted(
            os.listdir(f"data/862CT/res_data/origin_img/{mask_name}"),
            key=lambda x: int(x.split(".")[0].split("-")[1]),
            reverse=True,
        )

        # print(mask_img_list_small)
        # print(img_list_small)
        # exit()

        # print(mask_name)
        for img_name, mask_img_name in zip(img_list_small, mask_img_list_small):
            # print(img_name)
            # print(mask_img_name)
            img = cv2.imread(f"data/862CT/res_data/origin_img/{mask_name}/{img_name}")
            mask_img = cv2.imread(f"data/862CT/res_data/mask_img/{mask_name}/{mask_img_name}")
            if np.all(mask_img == 0):
                continue
            # print(img.shape)
            # print(mask_img_name)
            mask_img *= 255
            cv2.imwrite(f"data/res_data/origin/{mask_name}_{mask_img_name}", img)
            cv2.imwrite(f"data/res_data/mask/{mask_name}_{mask_img_name}", mask_img)
            # break
        # break


def adjust_image_orientation():
    mask_img_list = sorted(
        os.listdir("data/res_data/mask/"),
        key=lambda x: int(x.split(".")[0].split("_")[0].replace("V", "")),
    )
    for mask_img_name in mask_img_list:
        # print(mask_img_name)
        img = cv2.imread(f"data/res_data/mask/{mask_img_name}")
        #
        # # 转置图像
        img_transposed = cv2.transpose(img)
        #
        # # 保存或返回翻转后的图像
        cv2.imwrite(f"data/res_data/mask/{mask_img_name}", img_transposed)
        # # 或者直接显示
        # # cv2.imshow('Flipped Image', img_transposed)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # break


def get_coco_annotation(is_train=True):
    # image_paths = ['data/res_data/origin/1V_28.png', 'data/res_data/origin/1V_29.png', 'data/res_data/origin/1V_30.png']
    # mask_paths = ['data/res_data/mask/1V_28.png', 'data/res_data/mask/1V_29.png', 'data/res_data/mask/1V_30.png']
    # pass
    image_paths = sorted(os.listdir("data/res_data/origin/"))
    mask_paths = sorted(os.listdir("data/res_data/mask/"))

    # print(len(image_paths))
    if is_train:
        image_paths = image_paths[:2500]
        mask_paths = mask_paths[:2500]
    else:
        image_paths = image_paths[2500:]
        mask_paths = mask_paths[2500:]

    # for a,b in zip(image_paths, mask_paths):
    #     assert a == b
    # print(image_paths)
    # print(mask_paths)
    # exit()

    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "adca"}, {"id": 1, "name": "SRCC"}],  # 示例类别
    }

    label_df = pd.read_csv("data/res_data/label.csv")

    # 遍历图像和掩码，生成数据
    for img_id, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths), 1):
        # 读取图像和掩码
        image_path = "data/res_data/origin/" + image_path
        mask_path = "data/res_data/mask/" + mask_path

        # print(image_path)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        file_name = os.path.basename(image_path)
        # print(file_name)
        file_index = int(file_name.split(".")[0].split("_")[0].replace("V", ""))
        # 添加图像信息
        coco_dataset["images"].append(
            {
                "id": img_id,
                "width": image.shape[1],
                "height": image.shape[0],
                "file_name": file_name,
            }
        )

        # 假设每个掩码图只包含一个对象，计算边界框
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        x, y, w, h = cv2.boundingRect(contours[0])

        category_name = int(
            label_df[label_df["number"] == file_index]["label"].values[0]
        )
        # 添加标注信息
        coco_dataset["annotations"].append(
            {
                "id": img_id,
                "image_id": img_id,
                "category_id": category_name,  # 假设所有对象属于同一类别
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [],  # 可以根据需要添加
                "iscrowd": 0,
            }
        )

    # 保存为JSON文件
    if is_train:
        with open("data/res_data/ct_dataset_train.json", "w") as f:
            json.dump(coco_dataset, f, indent=4)
    else:
        with open("data/res_data/ct_dataset_val.json", "w") as f:
            json.dump(coco_dataset, f, indent=4)
    # with open('data/res_data/ct_dataset.json', 'w') as f:
    #     json.dump(coco_dataset, f, indent=4)


# 验证标注是否正确
def validate_annotation():
    from pycocotools.coco import COCO

    # import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    image_dir = "data/res_data/origin"  # 数据集目录
    # dataType = 'data/res_data/ct_dataset.json'  # 数据集类型
    annFile = "data/res_data/ct_dataset_train.json"
    image_id = 3  # 标注文件路径
    coco = COCO(annFile)

    # 通过image_id获取图片信息
    img_info = coco.loadImgs(image_id)[0]

    # 加载图片
    img_path = os.path.join(image_dir, img_info["file_name"])
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # 获取该图片的所有标注
    annIds = coco.getAnnIds(imgIds=img_info["id"], iscrowd=None)
    anns = coco.loadAnns(annIds)

    # 遍历每个标注
    for ann in anns:
        # 获取类别名
        cat = coco.loadCats(ann["category_id"])
        class_name = cat[0]["name"]

        # 获取边界框 [x,y,width,height]
        bbox = ann["bbox"]
        # 绘制边界框
        draw.rectangle(
            [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            outline="red",
            width=3,
        )

        # 添加类别名文字
        draw.text((bbox[0] + 10, bbox[1] + 10), class_name, fill="red")

    # 显示图片
    img.show()


if __name__ == "__main__":
    # get_png_data()
    # get_use_data()
    # adjust_image_orientation()
    # get_coco_annotation(is_train=True)
    validate_annotation()
