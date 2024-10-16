from datasets import load_dataset
import os
import csv
from PIL import Image
import io

def load_and_save_dataset(dataset_name, output_dir):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the dataset and save images
    for split in dataset.keys():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for idx, example in enumerate(dataset[split]):
            # Get the image and label
            image = example['image']
            
            # Generate a filename
            filename = f"{split}_{idx:05d}.jpg"
            file_path = os.path.join(split_dir, filename)
            
            # Save the image
            if isinstance(image, Image.Image):
                image.save(file_path)
            else:
                Image.open(io.BytesIO(image)).save(file_path)
    
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    dataset_name = "e1010101/tongue-images-384-segmented"
    output_dir = "Data/tongue_images_segmented"
    load_and_save_dataset(dataset_name, output_dir)
