import cv2
import os
import pandas as pd 

# origin_path = "res_data/mask/"
origin_path = "res_data/use_mask/"
origin_img_list = os.listdir(origin_path)

# label_df = pd.read_csv("res_data/label.csv")
label_df = pd.read_csv("res_data/201_label.csv")
# print(label_df.describe())
value_counts = label_df['label'].value_counts()
# print(value_counts)
# exit()

for img_name in origin_img_list:
    img = cv2.imread(os.path.join(origin_path,img_name))
    # print(img.shape)
    # break
    # img_v2 = img[:,:,0]
    img_v2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # import pdb
    # pdb.set_trace()
    # print(img_name)
    img_index = int(img_name.split(".")[0].split("_")[0].replace("V", ""))
    category_name = int(label_df[label_df["number"] == img_index]["label"].values[0])
    if category_name == 0:
        img_v2[img_v2 != 0] = 1
    else:
        img_v2[img_v2 != 0] = 2

    cv2.imwrite(os.path.join("res_data/use_mask_v2",img_name),img_v2)
    # break