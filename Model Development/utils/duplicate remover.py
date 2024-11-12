import os
from pathlib import Path

def remove_duplicates():
    # Define the directories
    annotated_dir = Path("dataset_annotated")
    all_dir = Path("dataset_unannotated")

    # Ensure both directories exist
    if not (annotated_dir.exists() and all_dir.exists()):
        print("Error: One or both directories do not exist")
        return

    # Get list of files in annotated directory
    annotated_files = set(file.name for file in annotated_dir.iterdir() if file.is_file())

    # Check files in 'all' directory and remove duplicates
    removed_count = 0
    for file in all_dir.iterdir():
        if file.is_file() and file.name in annotated_files:
            file.unlink()  # Delete the file
            removed_count += 1

    print(f"Removed {removed_count} duplicate files from 'all' directory")

if __name__ == "__main__":
    remove_duplicates()