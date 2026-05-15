import os

dataset_path = "final_dataset"

label_map = {
    "chickenpox": 0,
    "eczema": 1,
    "ringworm": 2
}

splits = ["train", "valid", "test"]

for split in splits:

    txt_file = open(f"{split}.txt", "w")

    split_path = os.path.join(dataset_path, split)

    for disease in os.listdir(split_path):

        disease_path = os.path.join(split_path, disease)

        label = label_map[disease]

        for image_name in os.listdir(disease_path):

            image_path = os.path.join(split, disease, image_name)

            txt_file.write(f"{image_path} {label}\n")

    txt_file.close()

print("TXT files created successfully")