import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np



input_folder = "dataset/eczema"
output_folder = "processed_dataset/eczema"

new_size = (224, 224)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

count = 1


def load_image_array(path):
    image = mpimg.imread(path)
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0

    return image


def resize_nearest(image, size):
    target_width, target_height = size
    height, width = image.shape[:2]
    row_indices = np.linspace(0, height - 1, target_height).astype(np.int64)
    col_indices = np.linspace(0, width - 1, target_width).astype(np.int64)
    return image[row_indices][:, col_indices]


for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    
    try:
        
        img = load_image_array(file_path)
        img = resize_nearest(img, new_size)
        
       
        new_name = f"eczema{count}.png"
        save_path = os.path.join(output_folder, new_name)
        
       
        plt.imsave(save_path, img)
        
        count += 1
        
    except Exception as e:
        print(f"Skipping {file_name}: {e}")

print("Done processing eczema images ✅")
