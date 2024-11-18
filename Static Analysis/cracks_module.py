import cv2
import numpy as np
import os
import math

from utils.visualization import *
from utils.util_functions import *

'''
This file detects cracks on a tongue image using canny and edge detection. 
Running the file will produce a visualization in the "Visualizations" directory.
'''

def detect_cracks(image, vertical_canny_threshold_low=30, vertical_canny_threshold_high=140, 
                        horizontal_canny_threshold_low=40, horizontal_canny_threshold_high=150, 
                        vertical_hough_threshold=30, horizontal_hough_threshold=75, 
                        min_line_length_vertical=50, max_line_gap_vertical=20,
                        min_line_length_horizontal=50, max_line_gap_horizontal=20,
                        dynamic_threshold=0, return_visualization=True):
    original_image = image.copy()
    cropped_image = crop_center(image)
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply vertical edge detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 60))
    vertical_edges = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, vertical_kernel)

    # Apply horizontal edge detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 3))
    horizontal_edges = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Apply Canny edge detection on edges seperately
    canny_vertical = cv2.Canny(vertical_edges, vertical_canny_threshold_low, vertical_canny_threshold_high)
    canny_horizontal = cv2.Canny(horizontal_edges, horizontal_canny_threshold_low, horizontal_canny_threshold_high)
    
    # Use Probabilistic Hough Line Transform to detect lines
    lines_vertical = cv2.HoughLinesP(canny_vertical, 3, np.pi/180, vertical_hough_threshold, 
                                     minLineLength=min_line_length_vertical, maxLineGap=max_line_gap_vertical)
    lines_horizontal = cv2.HoughLinesP(canny_horizontal, 3, np.pi/180, horizontal_hough_threshold, 
                                       minLineLength=min_line_length_horizontal, maxLineGap=max_line_gap_horizontal)
    
    height, width = cropped_image.shape[:2]
    left_boundary = width // 3
    right_boundary = 2 * width // 3
    line_coords = []
    
    num_cracks = 0
    if lines_vertical is not None:
        for line in lines_vertical:
            x1, y1, x2, y2 = line[0]
            # Check if the line is within the central third region of image
            if (left_boundary <= x1 <= right_boundary) and (left_boundary <= x2 <= right_boundary):
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Detect vertical lines
                if (abs(angle) > 80 and abs(angle) < 100):
                    if line_length >= min_line_length_vertical:
                        line_coords.append([x1, y1, x2, y2])
                        num_cracks += 1
                        
    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            # Check if the line is within the central third region of image
            if (left_boundary <= x1 <= right_boundary) and (left_boundary <= x2 <= right_boundary):
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Detect horizontal lines
                if (abs(angle) < 20 or abs(angle) > 150):
                    if line_length >= min_line_length_horizontal:
                        line_coords.append([x1, y1, x2, y2])
                        num_cracks += 1
                        
                    
    grid_visualization = None

    if return_visualization:
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        vertical_edges_rgb = cv2.cvtColor(vertical_edges, cv2.COLOR_GRAY2BGR)
        horizontal_edges_rgb = cv2.cvtColor(horizontal_edges, cv2.COLOR_GRAY2BGR)
        canny_vertical_rgb = cv2.cvtColor(canny_vertical, cv2.COLOR_GRAY2BGR)
        canny_horizontal_rgb = cv2.cvtColor(canny_horizontal, cv2.COLOR_GRAY2BGR)

        # Resize images to have the same height and width
        target_height, target_width = cropped_image.shape[0], cropped_image.shape[1]
        
        original_resized = cv2.resize(original_image, (target_width, target_height))
        cropped_resized = cv2.resize(cropped_image, (target_width, target_height))
        enhanced_resized = cv2.resize(enhanced_rgb, (target_width, target_height))
        hough_resized = cv2.resize(cropped_image, (target_width, target_height))
        vertical_edges_resized = cv2.resize(vertical_edges_rgb, (target_width, target_height))
        horizontal_edges_resized = cv2.resize(horizontal_edges_rgb, (target_width, target_height))
        canny_vertical_resized = cv2.resize(canny_vertical_rgb, (target_width, target_height))
        canny_horizontal_resized = cv2.resize(canny_horizontal_rgb, (target_width, target_height))
        
        for line in line_coords:
            x1, y1, x2, y2 = line
            cv2.line(hough_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Add labels to all images
        vis_original = add_label(original_resized, "Original")
        vis_cropped = add_label(cropped_resized, "Cropped")
        vis_enhanced = add_label(enhanced_resized, "Contrast Enhanced")
        vis_hough = add_label(hough_resized, "Hough Lines")
        vis_vertical_edges = add_label(vertical_edges_resized, "Vertical Edges")
        vis_horizontal_edges = add_label(horizontal_edges_resized, "Horizontal Edges")
        vis_canny_vertical = add_label(canny_vertical_resized, "Vertical Canny")
        vis_canny_horizontal = add_label(canny_horizontal_resized, "Horizontal Canny")
        vis_result = add_label(original_resized.copy(), "Result")
        cv2.putText(vis_result, f"Cracks: {num_cracks}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        top_row = np.hstack((vis_original, vis_cropped, vis_enhanced))
        row_2 = np.hstack((vis_vertical_edges, vis_horizontal_edges, np.zeros_like(vis_original)))
        row_3 = np.hstack((vis_canny_vertical, vis_canny_horizontal, vis_hough))
        bottom_row = np.hstack((vis_result, np.zeros_like(vis_original), np.zeros_like(vis_original)))

        grid_visualization = np.vstack((top_row, row_2, row_3, bottom_row))
        
    result = False
    if num_cracks >= dynamic_threshold:
        result = True
        
    return result, grid_visualization


def detect_cracks_dynamic(image):
    tongue_percentage = get_tongue_area_percentage(image)
    
    # print(tongue_percentage)
    
    vertical_canny_threshold_low = 0
    vertical_canny_threshold_high = 0
    horizontal_canny_threshold_low = 0
    horizontal_canny_threshold_high = 0
    vertical_hough_threshold = 0
    horizontal_hough_threshold = 0
    min_line_length_vertical = 0
    max_line_gap_vertical = 0
    min_line_length_horizontal = 0
    max_line_gap_horizontal = 0
    dynamic_threshold = 0 
       
    if tongue_percentage >= 40:
        vertical_canny_threshold_low = 50
        vertical_canny_threshold_high = 150
        horizontal_canny_threshold_low = 45
        horizontal_canny_threshold_high = 150
        vertical_hough_threshold = 30
        horizontal_hough_threshold = 75
        min_line_length_vertical = 30
        max_line_gap_vertical = 7
        min_line_length_horizontal = 60
        max_line_gap_horizontal = 7
        dynamic_threshold = 35
    elif tongue_percentage >= 20:
        vertical_canny_threshold_low = 37.5
        vertical_canny_threshold_high = 135
        horizontal_canny_threshold_low = 45
        horizontal_canny_threshold_high = 140
        vertical_hough_threshold = 20
        horizontal_hough_threshold = 65
        min_line_length_vertical = 20
        max_line_gap_vertical = 6
        min_line_length_horizontal = 55
        max_line_gap_horizontal = 8
        dynamic_threshold = 20
    else:
        vertical_canny_threshold_low = 35
        vertical_canny_threshold_high = 130
        horizontal_canny_threshold_low = 40
        horizontal_canny_threshold_high = 130
        vertical_hough_threshold = 10
        horizontal_hough_threshold = 60
        min_line_length_vertical = 15
        max_line_gap_vertical = 2
        min_line_length_horizontal = 50
        max_line_gap_horizontal = 9
        dynamic_threshold = 15

    return detect_cracks(image, vertical_canny_threshold_low, vertical_canny_threshold_high, 
                         horizontal_canny_threshold_low, horizontal_canny_threshold_high,
                         vertical_hough_threshold, horizontal_hough_threshold, 
                         min_line_length_vertical, max_line_gap_vertical,
                         min_line_length_horizontal, max_line_gap_horizontal, 
                         dynamic_threshold, return_visualization=True)


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
            
    #         num_cracks, visualization = detect_cracks_with_visualization(image)

    #         # Save visualization
    #         vis_path = 'Test Images/Crack Visualizations/test_visualization_cracks' + image_file
    #         cv2.imwrite(vis_path, visualization)
            
    # Test single image
    image_path = 'Samples/sample_7.jpg'
    image = cv2.imread(image_path)
    result, visualization = detect_cracks_dynamic(image)
    cv2.imwrite('Visualizations/test_visualization_cracks_single.jpg', visualization)
    print(result)
    
if __name__ == "__main__":
    main()
