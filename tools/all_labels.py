#修改图片/标签的名称
import os

#训练集数据
img_path = "../Dataset/train/val_candle/val/images"
file = [f for f in os.listdir(img_path) if f.endswith(".png")]

#图像名称修改
for i,filename in enumerate(file,start=2091):
    old_path = os.path.join(img_path, filename)
    new_name = f"000000{i:}.png"
    new_path = os.path.join(img_path, new_name)
    os.replace(old_path, new_path)


folder_label_path = '../Dataset/train/val_candle/val/labels'
label_files = [f for f in os.listdir(folder_label_path) if f.endswith(".txt")]

#标签名称修改
for j,labelname in enumerate(label_files,start=2091):
    old_path = os.path.join(folder_label_path, labelname)
    new_name_label = f"000000{j:}.txt"
    new_path = os.path.join(folder_label_path, new_name_label)
    os.replace(old_path, new_path)

