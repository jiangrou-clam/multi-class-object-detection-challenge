#过采样与过程验证脚本
#支持类别数量计算、图像水平翻转、检测框绘制、批量补充图像标签的功能

import os
import cv2

result = []

def candle_check(path):
    for i in os.listdir(path):
        p = os.path.join(path, i)
        with open(p, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "2" in line[0]:
                    result.append(i)
    return print(f"candle number:{len(result)},candle list 5th content:{result[:5]}")

# img = cv2.imread("../Dataset/1/0000002091.png")
def img_flip(img):
    aug_img = cv2.flip(img, 1)
    return aug_img

# output_path = "../Dataset/1/candle_001_flip.txt"
def label_calculater(label):
    content = []
    with open(label, "r") as f:
        lines = f.readlines()
        for line in lines:
            p = line.strip().split()
            x_center = 1 - float(p[1])
            n_p = f"{p[0]} {x_center:.20f} {p[2]} {p[3]} {p[4]}\n"
            content.append(n_p)
    return content




# label = "../Dataset/1/0000002091.txt"


def box_check(label,aug_img):
    content = []
    height, width = aug_img.shape[:2]
    with open(label, "r") as f:
        lines = f.readlines()
        for line in lines:
            p = line.strip().split()
            x_center = 1 - float(p[1])
            n_p = f"{p[0]} {x_center:.20f} {p[2]} {p[3]} {p[4]}\n"
            content.append(n_p)
            y_center = float(p[2])
            box_w = float(p[3])
            box_h = float(p[4])
            x1 = int((x_center - box_w / 2) * width)
            y1 = int((y_center - box_h / 2) * height)
            x2 = int((x_center + box_w / 2) * width)
            y2 = int((y_center + box_h / 2) * height)
            cv2.rectangle(aug_img, (x1, y1), (x2, y2), (0, 255, 0),5)
    return cv2.imwrite(f"../Dataset/1/filp_check.png", aug_img)


img_path = "../Dataset/1/val/images/"
label_path = "../Dataset/1/val/labels/"
output_img = "../Dataset/1/aug/images/"
output_label = "../Dataset/1/aug/labels/"

# for f in os.listdir(img_path):
#     root = os.path.join(img_path,f)
#     img = cv2.imread(root)
#     process_img = img_flip(img)
#     stem = os.path.splitext(f)[0]
#     for i in range(8):
#         output_path_aug_img = os.path.join(output_img, f"{stem}_aug_copy_0{i}.png")
#         output_path_img = os.path.join(output_img,f"{stem}_copy_0{i}.png")
#         cv2.imwrite(output_path_aug_img, process_img)
#         cv2.imwrite(output_path_img, img)
# print("img has all completed")
#
# for f in os.listdir(label_path):
#     root = os.path.join(label_path,f)
#     with open(root,"r") as p:
#         label = p.readlines()
#     process_label = label_calculater(root)
#     stem = os.path.splitext(f)[0]
#     for i in range(8):
#         output_path_aug_label = os.path.join(output_label, f"{stem}_aug_copy_0{i}.txt")
#         output_path_label = os.path.join(output_label, f"{stem}_copy_0{i}.txt")
#         with open(output_path_aug_label, "w") as p:
#             p.writelines(process_label)
#         with open(output_path_label, "w") as p:
#             p.writelines(label)
# print("label has all completed")
#

path = "../Dataset/train/labels"
candle_check(path)

# aug_img = cv2.imread("../Dataset/1/aug/images/000000009_aug_copy_01.png")
# aug_label = "../Dataset/1/aug/labels/000000009_copy_01.txt"
# box_check(aug_label,aug_img)















