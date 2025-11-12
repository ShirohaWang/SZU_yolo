import cv2
import albumentations as A
import os
from tqdm import tqdm

#数据路径
input_dir = "dataset_split/images/test"
output_dir = "dataset_split/images/test_aug"
os.makedirs(output_dir, exist_ok=True)

# 定义增强操作
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.3),
    A.Rotate(limit=10, p=0.5)
])

#遍历增强所有图片
for img_name in tqdm(os.listdir(input_dir)):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    aug = augmentor(image=img)
    cv2.imwrite(os.path.join(output_dir, img_name), aug["image"])

print("数据增强完成，增强后的图片保存在：", output_dir)
