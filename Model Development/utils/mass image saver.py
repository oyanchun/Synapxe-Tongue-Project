import pandas as pd
import shutil
import os

# Change as needed
df = pd.read_csv('image_clusters.csv')
base_dir = 'clustered_images'
os.makedirs(base_dir, exist_ok=True)

for index, row in df.iterrows():
    filename = row['filename']
    cluster = row['cluster']
    
    # Create a directory for the cluster if it doesn't exist
    cluster_dir = os.path.join(base_dir, f'cluster_{cluster}')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Construct the source and destination paths
    source_path = filename
    image_name = filename.split("\\")[-1]
    dest_path = os.path.join(cluster_dir, image_name)
    
    # Copy the image to the appropriate cluster directory
    shutil.copy2(source_path, dest_path)