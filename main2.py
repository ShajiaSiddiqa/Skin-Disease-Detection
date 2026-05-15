import os
import shutil
import random


source_dir = "processed_dataset"
dest_dir = "final_dataset"

train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1


for split in ["train", "valid", "test"]:
    for class_name in os.listdir(source_dir):
        path = os.path.join(dest_dir, split, class_name)
        os.makedirs(path, exist_ok=True)


for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train_imgs = images[:train_end]
    valid_imgs = images[train_end:valid_end]
    test_imgs = images[valid_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(dest_dir, "train", class_name, img))

    for img in valid_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(dest_dir, "valid", class_name, img))

    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(dest_dir, "test", class_name, img))

print("Dataset split complete ✅")