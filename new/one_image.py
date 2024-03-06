from rich import print
from mmdet.apis import init_detector, inference_detector

# use shell
# python new/image_demo.py data/res_data/origin/2V_17.png 
# new/configs/rtmdet/model_config.py 
# --weights work_dirs/model_config/best_coco_bbox_mAP_epoch_18.pth 
# --device cpu

def get_one_image_laebl(image_name):
    config_file = "new/configs/rtmdet/model_config.py"
    checkpoint_file = "work_dirs/model_config/best_coco_bbox_mAP_epoch_30.pth"
    model = init_detector(config_file, checkpoint_file, device="cpu")  # or device='cuda:0'
    infer_res = inference_detector(model, "data/res_data/origin/2V_18.png")

    # print(infer_res.pred_instances.labels)
    # print(infer_res.pred_instances.scores)
    label = infer_res.pred_instances.labels[0].item()
    score = infer_res.pred_instances.scores[0].item()

    print(label)
    print(score)
    return label
