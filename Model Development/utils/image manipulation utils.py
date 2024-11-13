import numpy as np
from PIL import Image
import cv2

def crop_center_10_percent(image):
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    
    # Calculate the dimensions of the 40% crop
    crop_height = int(height * 0.1)
    crop_width = int(width * 0.1)
    
    # Calculate the starting points for cropping
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2
    
    # Crop the image
    cropped_image_array = image_array[start_y:start_y+crop_height, start_x:start_x+crop_width]
    
    # Convert back to PIL Image
    cropped_image = Image.fromarray(cropped_image_array)
    
    return cropped_image


def crop_dataset(dataset):
    cropped_dataset = dataset.map(lambda example: {'image': crop_center_10_percent(example['image'])})
    return cropped_dataset


def create_edge_mask(image, tongue_mask):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Find the contour of the tongue
    contours, _ = cv2.findContours(tongue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        tongue_contour = max(contours, key=cv2.contourArea)
        
        # Create a slightly smaller contour to focus on the edges
        epsilon = 0.02 * cv2.arcLength(tongue_contour, True)
        smaller_contour = cv2.approxPolyDP(tongue_contour, epsilon, True)
        
        # Draw the smaller contour on the mask
        cv2.drawContours(mask, [smaller_contour], 0, 255, thickness=int(width * 0.15))
    
    # crop top region
    mask[:int(height * 0.3), :] = 0
    
    return mask


def extract_edges(image):
    _, tongue_mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    edge_mask = create_edge_mask(image, tongue_mask)

    # Apply the edge mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=edge_mask)
    return masked_image