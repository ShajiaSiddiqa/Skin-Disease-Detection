import numpy as np
import matplotlib.image as mpimg


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


def resize_nearest(image, image_size):
    height, width = image.shape[:2]
    target_height, target_width = image_size
    row_indices = np.linspace(0, height - 1, target_height).astype(np.int64)
    col_indices = np.linspace(0, width - 1, target_width).astype(np.int64)
    return image[row_indices][:, col_indices]

def load_dataset(txt_file, dataset_root, image_size=(224, 224)):

    images = []
    labels = []

    with open(txt_file, "r") as file:

        for line in file:

            path, label = line.strip().split()

            full_path = dataset_root + "/" + path

            image = load_image_array(full_path)
            image = resize_nearest(image, image_size)
            image = np.ascontiguousarray(image, dtype=np.float32)

            images.append(image)

            labels.append(int(label))

    X = np.array(images)
    y = np.array(labels)

    return X, y


X_train, y_train = load_dataset(
    "train.txt",
    "final_dataset"
)

X_valid, y_valid = load_dataset(
    "valid.txt",
    "final_dataset"
)

X_test, y_test = load_dataset(
    "test.txt",
    "final_dataset"
)

print(X_train.shape)
