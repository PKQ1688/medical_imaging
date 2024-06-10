import torch
import pandas as pd
from PIL import Image
import os
import torch.nn.functional as F
from transformers import MaskFormerFeatureExtractor,MaskFormerForInstanceSegmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "use_huggingface/final_checkpoint"

feature_extractor = MaskFormerFeatureExtractor.from_pretrained(model_name)
# inputs = feature_extractor(images=image, return_tensors="pt")
model = MaskFormerForInstanceSegmentation.from_pretrained(model_name).to(device)

label_df = pd.read_csv("test_label.csv")

_image_name = "9v"  
img_paths = os.listdir(os.path.join("test_data_origin",_image_name))
img_paths = [os.path.join("test_data_origin",_image_name,img_name) for img_name in img_paths]
images = [Image.open(img_path) for img_path in img_paths]
print(len(images[0].getbands()))


for image in images:
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    pred_segs = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

print(pred_segs.shape)