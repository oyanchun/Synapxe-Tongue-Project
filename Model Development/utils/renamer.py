import os
import re

folder_path = 'dataset_annotated'
pattern = re.compile(r'^(CC_\d+)')

for filename in os.listdir(folder_path):
    old_file_path = os.path.join(folder_path, filename)
    
    if not os.path.isfile(old_file_path):
        continue
    match = pattern.match(filename)
    if match:
        new_filename = f"{match.group(1)}{os.path.splitext(filename)[1]}"
        new_file_path = os.path.join(folder_path, new_filename)
        
        os.rename(old_file_path, new_file_path)
        print(f'Renamed "{filename}" to "{new_filename}"')
    else:
        print(f'No matching pattern found in "{filename}", skipping.')
