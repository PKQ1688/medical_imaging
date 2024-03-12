import os

import pandas as pd
from mmdet.apis import init_detector, inference_detector
from rich import print

# use shell
# python new/image_demo.py data/res_data/origin/2V_17.png 
# new/configs/rtmdet/model_config.py 
# --weights work_dirs/model_config/best_coco_bbox_mAP_epoch_18.pth 
# --device cpu

config_file = "new/configs/rtmdet/model_config.py"
checkpoint_file = "work_dirs/model_config/best_coco_bbox_mAP_epoch_30.pth"
model = init_detector(config_file, checkpoint_file, device="cpu")


def get_one_image_laebl(image_name="2V_18.png"):
    # or device='cuda:0'
    infer_res = inference_detector(model, f"data/res_data/origin/{image_name}")

    # print(infer_res.pred_instances.labels)
    # print(infer_res.pred_instances.scores)
    label = infer_res.pred_instances.labels[0].item()
    score = infer_res.pred_instances.scores[0].item()

    # print(label)
    # print(score)
    return label


def handle_test_res():
    image_paths = sorted(os.listdir("data/res_data/origin/"))
    image_paths = image_paths[2500:]

    # print(image_paths)
    res_dict = {image_name.split("_")[0].replace("V", ""): [] for image_name in image_paths}
    # print(res_dict)
    for image_name in image_paths:
        label = get_one_image_laebl(image_name)
        # print(image_name)
        # print(label)
        res_dict[image_name.split("_")[0].replace("V", "")].append(label)
        # break
    # print(res_dict)
    res_dict = {k: max(v, key=v.count) for k, v in res_dict.items()}
    print(res_dict)
    print(len(res_dict))

    label_df = pd.read_csv("data/res_data/label.csv")
    acc_count = 0
    for img_id, label in res_dict.items():
        if label == label_df.loc[label_df["number"] == int(img_id), "label"].item():
            acc_count += 1

    print("accuarcy: ", acc_count / len(res_dict))


if __name__ == '__main__':
    # get_one_image_laebl()
    handle_test_res()
