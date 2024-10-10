import cv2
import numpy as np
import os

from utils.visualization import *


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
    mask[:int(height * 0.3), :] = 0  # Remove top 20%
    
    return mask


def detect_toothmarks_by_color(image, edge_mask):
    # Apply the edge mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=edge_mask)
    
    # Convert masked image to LAB color space
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    
    l, _, _ = cv2.split(lab)
    
    # Apply bilateral filter to reduce noise while preserving edges
    l_filtered = cv2.bilateralFilter(l, 9, 75, 75)
    block_size = 41
    C = 3
    binary = cv2.adaptiveThreshold(l_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, block_size, C)
    
    # Apply the edge mask again to ensure we only keep the relevant areas
    binary = cv2.bitwise_and(binary, binary, mask=edge_mask)
    
    # Remove small noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return binary


def detect_toothmarks(image, return_visualization=True):
    vis_original = image.copy()
    
    # Create a mask for the tongue (non-black areas)
    _, tongue_mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vis_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    edge_mask = create_edge_mask(image, tongue_mask)
    
    vis_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
    masked_gray = cv2.bitwise_and(gray, gray, mask=edge_mask)
    vis_masked = cv2.cvtColor(masked_gray, cv2.COLOR_GRAY2BGR)
    non_zero_mask = cv2.threshold(masked_gray, 1, 255, cv2.THRESH_BINARY)[1]

    block_size = 35  # Size of the local neighborhood for thresholding
    C = 2  # Differences in color
    dark_spots = cv2.adaptiveThreshold(masked_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, block_size, C)
    dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=non_zero_mask)
    dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=tongue_mask)
    
    height, width = dark_spots.shape
    dark_spots[:int(height * 0.4), :] = 0
    
    vis_threshold = cv2.cvtColor(dark_spots, cv2.COLOR_GRAY2BGR)
    color_based_spots = detect_toothmarks_by_color(image, edge_mask)
    combined_spots = cv2.bitwise_or(dark_spots, color_based_spots)
    vis_color_based = cv2.cvtColor(color_based_spots, cv2.COLOR_GRAY2BGR)
    
    # Find and draw contours
    contours, _ = cv2.findContours(combined_spots, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    vis_contours = image.copy()
    cv2.drawContours(vis_contours, contours, -1, (0, 255, 0), 2)
    
    valid_spots = 0
    vis_filtered = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if 150 < area < 1000:
            cv2.drawContours(vis_filtered, [contour], 0, (0, 0, 255), 2)
            valid_spots += 1
    
    result = True if valid_spots >= 1 else False
    
    grid_visualization = None
    
    if return_visualization:
        vis_original = add_label(vis_original, 'Original')
        vis_gray = add_label(vis_gray, 'Gray')
        vis_masked = add_label(vis_masked, 'Masked')
        vis_threshold = add_label(vis_threshold, 'Threshold')
        vis_contours = add_label(vis_contours, 'Contours')
        vis_filtered = add_label(vis_filtered, 'Filtered')
        vis_color_based = add_label(vis_color_based, 'Color Based')
        vis_result = add_label(image.copy(), 'Result')
        cv2.putText(vis_result, f"Toothmarks: {result}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        top_row = np.hstack((vis_original, vis_gray, vis_masked))
        middle_row = np.hstack((vis_threshold, vis_contours, vis_filtered))
        bottom_row = np.hstack((vis_color_based, vis_result, np.zeros_like(vis_original)))
        
        grid_visualization = np.vstack((top_row, middle_row, bottom_row))
    
    return result, grid_visualization


def main():
    # Test on suite of images
    # image_path = 'Test Images'
    # images = os.listdir(image_path)
    # for image_file in images:
    #     image_path = 'Test Images'
    #     if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         image_path = os.path.join(image_path, image_file)
    #         image = cv2.imread(image_path)
    #         if image is None:
    #             print(f"Error: Unable to read image at {image_path}")
    #             continue
            
    #         num_toothmarks, visualization = detect_toothmarks_with_visualization(image)

    #         # Save visualization
    #         vis_path = 'Test Images/Toothmark Visualizations/test_visualization_toothmarks' + image_file
    #         cv2.imwrite(vis_path, visualization)
            
    # Test single image
    image_path = 'Samples/sample_1.jpg'
    image = cv2.imread(image_path)
    num_toothmarks, visualization = detect_toothmarks(image)
    cv2.imwrite('Visualizations/test_visualization_toothmarks_single.jpg', visualization)
    
if __name__ == "__main__":
    main()