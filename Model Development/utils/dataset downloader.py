from datasets import load_dataset
from tqdm import tqdm

import os
import csv

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.getenv('HF_KEY'))

# Change as needed
dataset = load_dataset("e1010101/tongue-images-384-segmented")

def save_dataset(data_split, output_dir):
    split_str = output_dir.split("_")[-1]
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata = []
    for i, example in enumerate(tqdm(data_split, desc=f"Saving {output_dir}")):
        image = example['image']
        labels = example['labels']
        image_filename = f"{split_str}_image_{i:05d}.png"
        image.save(os.path.join(images_dir, image_filename))
        metadata.append({"filename": image_filename, "labels": labels})
    
    with open(os.path.join(output_dir, "metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "labels"])
        writer.writeheader()
        writer.writerows(metadata)

save_dataset(dataset["train"], "downloaded_tongue_images_train")
save_dataset(dataset["validation"], "downloaded_tongue_images_valid")
save_dataset(dataset["test"], "downloaded_tongue_images_test")

print("Dataset saved successfully!")