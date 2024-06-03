import torch
import pandas as pd
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "segformer-b0-finetuned-segments-sidewalk-outputs/checkpoint-7240"
feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)

label_df = pd.read_csv("test_label.csv")
# img_path = 'res_data/origin/1V_28.png'
def get_pred(img_paths):
    images = [Image.open(img_path) for img_path in img_paths]
    inputs = feature_extractor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits

    upsampled_logits = F.interpolate(
        logits,
        size=images[0].size[::-1],  # Assuming all images are of the same size
        mode='bilinear',
        align_corners=False
    )

    pred_segs = upsampled_logits.argmax(dim=1)

    results = []
    for pred_seg in pred_segs:
        flat_tensor = pred_seg.view(-1)
        
        # Count the occurrences of each class
        unique, counts = torch.unique(flat_tensor, return_counts=True)
        counts_dict = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))
        
        # Get the counts of classes 0, 1, and 2
        zeros = counts_dict.get(0, 0)
        ones = counts_dict.get(1, 0)
        twos = counts_dict.get(2, 0)
        
        # Determine the result based on counts of classes 1 and 2
        if ones > twos:
            results.append(0)
        else:
            results.append(1)
    
    return results

img_list = os.listdir('res_data/origin')
img_paths = [os.path.join('res_data/origin', img) for img in img_list]
print(f"Total images: {len(img_paths)}")

# Example of processing in batches
# batch_size = 16
predictions = []
labels = []
for i in tqdm(range(0, len(img_paths), batch_size),total=int(len(img_paths)//batch_size)):
    batch_paths = img_paths[i:i+batch_size]
    # import pdb
    # pdb.set_trace()
    batch_labels = []
    for image_name in batch_paths:
        img_name = os.path.basename(image_name)
        img_index = int(img_name.split(".")[0].split("_")[0].replace("V", ""))
        category_name = int(label_df[label_df["number"] == img_index]["label"].values[0])
        batch_labels.append(category_name)
    predictions.extend(get_pred(batch_paths))
    labels.extend(batch_labels)

print(predictions)
print(labels)

# 计算各项指标
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

# 打印结果
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")