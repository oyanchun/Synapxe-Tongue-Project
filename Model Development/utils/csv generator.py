import os
import pandas as pd

dataset_path = "dataset_unannotated"

image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

df = pd.DataFrame(image_files, columns=['filename'])

df.to_csv('unannotated_images.csv', index=False)

print(f"Generated CSV file with {len(image_files)} image filenames")