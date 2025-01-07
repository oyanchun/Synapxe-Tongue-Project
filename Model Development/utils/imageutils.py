import numpy as np
from PIL import Image
import cv2

def crop_center_10_percent(image):
    '''
    Crops out only the centre 10% of an image.

    Args:
        image (PIL.Image): image to crop

    Returns:
        cropped_image (PIL.Image): image after cropping
    '''
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
    '''
    Crops images in the dataset.

    Args:
        dataset (datasets.Dataset): dataset to crop the images from.

    Returns:
        cropped_dataset (datasets.Dataset): dataset with its images cropped to 10% in the centre
    '''
    cropped_dataset = dataset.map(lambda example: {'image': crop_center_10_percent(example['image'])})
    return cropped_dataset


def create_edge_mask(image, tongue_mask):
    '''
    Creates an edge mask for the tongue region in an image.

    This function identifies the largest contour from the provided tongue mask,
    approximates a slightly smaller version of the contour to focus on the edges,
    and draws the smaller contour on a new mask. Additionally, the top region
    of the mask is cropped to exclude irrelevant areas.

    Args:
        image (numpy.ndarray): 
            The original image (in BGR format) from which the tongue was detected.
        tongue_mask (numpy.ndarray): 
            A binary mask where the tongue region is marked as white (255) 
            and the background as black (0).

    Returns:
        mask (numpy.ndarray): 
            A binary mask (same dimensions as the input image) with the edges of the tongue highlighted.

    Raises:
        ValueError: If no contours are found in the tongue mask.
    '''
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
    '''
    Extracts edges from an image and highlights them using an edge mask.

    Args:
        image (numpy.ndarray): 
            The input image in BGR format.

    Returns:
        masked_image (numpy.ndarray): 
            The output image where only the edges are highlighted, and the 
            rest of the image is blacked out.
    '''
    _, tongue_mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    edge_mask = create_edge_mask(image, tongue_mask)

    # Apply the edge mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=edge_mask)
    return masked_image