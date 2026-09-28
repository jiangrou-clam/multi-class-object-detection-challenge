import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import shutil
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import albumentations as A
from sahi import AutoDetectionModel
from sahi.predict import get_prediction

train_images = Path('Dataset/train/images')
train_masks = Path('Dataset/train/labels')

test_images = Path("testImages/testImages/images")

output_dir = Path("kaggle/predictions/labels")
output_dir.mkdir(parents=True, exist_ok=True)


def visualize_random_masks(images_dir, masks_dir, counts=5):
    """
    Vizualize random images with masks.

    images_dir: path to directory with images
    masks_dir: path to directory with masks
    counts: Numbers of images to show
    """

    image_paths = list(images_dir.glob("*"))
    samples = np.random.choice(image_paths, counts, replace=False)

    plt.figure(figsize=(counts * 5, 8))

    for i, img_path in enumerate(samples, 1):
        mask_path = masks_dir / img_path.with_suffix(".txt").name

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        with open(mask_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            try:
                class_id, x_center, y_center, box_w, box_h = map(float, parts)
            except:
                class_id, confidence, x_center, y_center, box_w, box_h = map(float, parts)

            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        plt.subplot(1, counts, i)
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()


transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


data_yaml = "./yolo_params.yaml"


if __name__ == '__main__':
    model = YOLO("./yolo11n.pt")

    results = model.train(
        data=data_yaml,
        lr0=0.005,
        epochs=150,
        batch=16,
        imgsz=640,
        device=0,
        optimizer="AdamW",
        seed=42,
        workers=0,
        mosaic=1.0,
        copy_paste=0.2
    )

    model = YOLO(r"runs\detect\train\weights\best.pt")
    print("predict is starting now")

    for img_path in test_images.glob("*"):
        results = model.predict(img_path, conf=0.05, device=0, verbose=False)  # 0 - GPU or "cpu"
        output_txt = output_dir / f"{img_path.stem}.txt"

        with open(output_txt, "w") as f:
            found = False
            for result in results:
                img_height, img_width = result.orig_shape
                boxes = result.boxes.data

                if boxes is None or len(boxes) == 0:
                    continue

                filtered_boxes = boxes[boxes[:, 4] >= 0.05]
                if len(filtered_boxes) == 0:
                    continue

                found = True
                for box in filtered_boxes:
                    x1, y1, x2, y2, confidence, cls_id = box.tolist()

                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    cls = int(cls_id)

                    f.write(f"{cls} {confidence:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n") #断点打在这

            if not found:
                f.write("")

    visualize_random_masks(test_images, output_dir)
    plt.show()

#sahi

    rows = []
    test_imgs = {p.stem for p in test_images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    predicted = set()

    for file in output_dir.glob("*.txt"):
        name = file.stem
        predicted.add(name)

        try:
            lines = [l.strip() for l in open(file) if len(l.strip().split()) == 6]
        except:
            lines = []

        rows.append({"image_id": name, "prediction_string": " ".join(lines) if lines else "no boxes"})

    for name in test_imgs - predicted:
        rows.append({"image_id": name, "prediction_string": "no boxes"})

    rows = pd.DataFrame(rows)
    rows.to_csv("submission.csv", index=False)
