#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/25 18:05
# @Author       : adolf
# @Email        : adolf1321794021@gmail.com
# @LastEditTime : 2024/5/25 18:05
# @File         : semantic_segmentation.py
import os
from PIL import Image

import numpy as np
import glob

# import json
import pdb

import torch
from torch import nn
from torchvision.transforms import ColorJitter
# from transformers import SegformerImageProcessor
# from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments, Trainer
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation

# from huggingface_hub import hf_hub_download

# from transformers import AutoFeatureExtractor

from mean_iou import MeanIoU
from datasets import Dataset  # , load_dataset
from collections import Counter


def load_images_and_masks(data_dir):
    """
    从给定的目录加载图片和对应的seg图片。

    :param data_dir: 包含图片和seg图片的目录路径
    :return: 两个列表，分别包含所有图片的numpy数组和对应seg图片的numpy数组
    """
    images_list = []
    seg_list = []

    images = sorted(glob.glob(os.path.join(data_dir, "origin", "*.png")))
    segs = sorted(glob.glob(os.path.join(data_dir, "mask_v2", "*.png")))

    for img_path in images:
        img = Image.open(img_path)
        images_list.append(img)

    for seg_path in segs:
        seg = Image.open(seg_path)
        seg_list.append(seg)

    return images_list, seg_list


data_dir = "res_data/"  # 替换为你的图片目录
images, masks = load_images_and_masks(data_dir)


# 将数据转换为Hugging Face的Dataset对象
data_dict = {"pixel_values": images, "label": masks}
dataset = Dataset.from_dict(data_dict)

# hf_dataset_identifier = "segments/sidewalk-semantic"
# dataset = load_dataset("segments/sidewalk-semantic")

# pdb.set_trace()
# exit()

dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.1)
train_ds = dataset["train"]
test_ds = dataset["test"]
# train_ds = dataset
# test_ds = dataset

# exit()

model_checkpoint = "maskformer-swin-base-ade"
id2label = {0: "background", 1: "adca", 2: "SRCC"}
label2id = {"background": 0, "adca": 1, "SRCC": 2}

# filename = "id2label.json"
# id2label = json.load(
#     open(hf_hub_download(hf_dataset_identifier, filename, repo_type="dataset"), "r")
# )
# id2label = {int(k): v for k, v in id2label.items()}
# label2id = {v: k for k, v in id2label.items()}

# id2label = {0: "background", 1: "frontground"}
# label2id = {"background": 0, "frontground": 1}

# feature_extractor = SegformerImageProcessor()
feature_extractor = MaskFormerFeatureExtractor.from_pretrained(model_checkpoint)
# feature_extractor = SegformerImageProcessor.from_pretrained(model_checkpoint)
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
# Define the color jitter transform
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels)
    return inputs


train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

# model = SegformerForSemanticSegmentation.from_pretrained(
#     model_checkpoint,
#     num_labels=len(id2label),
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True,  # Will ensure the segmentation specific components are reinitialized.
# )
model = MaskFormerForInstanceSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True, 
    )

epochs = 50
lr = 2e-4
batch_size = 4

training_args = TrainingArguments(
    "segformer-b0-finetuned-segments-sidewalk-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    eval_strategy="steps",
    # eval_strategy="no",
    save_strategy="steps",
    save_steps=400,
    eval_steps=400,
    logging_steps=1,
    # eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=False,
    hub_strategy="end",
)

metric = MeanIoU()


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # import pdb
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            # reduce_labels=feature_extractor.reduce_labels,
            reduce_labels=feature_extractor.do_reduce_labels,
        )

        correct = 0
        evel_num = pred_labels.shape[0]

        # Iterate over each sample
        for i in range(evel_num):
            values_pred, counts_pred = np.unique(pred_labels[i], return_counts=True)
            values_label, counts_label = np.unique(labels[i], return_counts=True)

            if len(values_pred) == 2:
                if values_pred[1] == values_label[1]:
                    correct += 1
            elif len(values_pred) == 1:
                continue
            else:
                try:
                    if counts_pred[1] >= counts_pred[2] and values_label[1] == 1:
                        correct += 1
                    if counts_pred[2] >= counts_pred[1] and values_label[1] == 2:
                        correct += 1
                except IndexError:
                    pass  # Handle the case where the array might be out of bounds
                    print("IndexError: Index out of bounds")
        
        accuracy = correct / evel_num
        metrics["accuracy"] = accuracy
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update(
            {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
        )
        metrics.update(
            {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)}
        )

        return metrics


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=feature_extractor,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    # compute_metrics=compute_metrics,
)

trainer.train()

output_dir = "final_checkpoint"
trainer.model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)