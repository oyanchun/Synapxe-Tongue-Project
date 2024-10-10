import numpy as np

def crop_center(image):
    height, width = image.shape[:2]
    
    # Find the bounding box of the non-black area
    non_black = np.where(image != 0)
    top, bottom = non_black[0].min(), non_black[0].max()
    left, right = non_black[1].min(), non_black[1].max()
    
    # Add a small margin to avoid cutting too close to the edge
    margin = 5
    top = max(0, top - margin)
    bottom = min(height, bottom + margin)
    left = max(0, left - margin)
    right = min(width, right + margin)
    
    # Crop the image
    return image[top:bottom, left:right]