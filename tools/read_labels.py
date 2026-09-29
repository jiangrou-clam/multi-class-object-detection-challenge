#批量读取标签的物体检测类型
#输出控制台上的标签名称与结果

import os

path = "../Dataset/val/labels"

file = os.listdir(path)
result = []

for i in file:
    p = os.path.join(path,i)
    with open(p,"r") as f:
        label = f.read(1)
        result.append(label)
        if label != "1" and label != "0":
            print(f"candle label is appearance,and the contect is {label}, name is {i}")
        else:
            continue
print(f"file is already processed, and label is:{result}")
