import cv2
import numpy as np

from utils.visualization import *
from utils.util_functions import *


def contour_near_edge(contour, image):
    """
    Check if the contour is near the edge of the image.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return x < 35 or y < 35 or x + w > image.shape[1] - 35 or y + h > image.shape[0] - 35


def detect_red_dots(image, circularity_min=0.2, return_visualization=True):
    """
    Detects and visualizes red dots on a tongue image.

    Args:
    image: The input image (segmented tongue against a black background).

    Returns:
    A tuple containing the number of red dots and a grid visualization of the detection process.
    """
    original_image = image.copy()
    tongue_percentage = get_tongue_area_percentage(image)
    original_image = crop_center(image)
    
    if tongue_percentage >= 40:
        area_min = 45
        area_max = 100
    elif tongue_percentage >= 20:
        area_min = 35
        area_max = 85
    else:
        area_min = 20
        area_max = 70
        
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # enhance image to increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Use adaptive thresholding to detect dark spots
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 4)


    kernel = np.ones((3,3), np.uint8)
    
    # MORPH_OPEN removes small noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # MORPH_CLOSE fills small holes
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_dots = 0
    
    # Create visualizations for grid
    vis_original = add_label(original_image.copy(), "Original")
    vis_gray = add_label(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale")
    vis_enhanced = add_label(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), "Enhanced")
    vis_blurred = add_label(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), "Blurred")
    vis_thresh = add_label(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "Thresholded")
    vis_contours = add_label(original_image.copy(), "All Contours")
    vis_filtered = add_label(original_image.copy(), "Filtered Contours")
    grid_visualization = None
    
    for contour in contours:
        # Ignore contour that are too near the edge of tongue
        if contour_near_edge(contour, original_image):
            continue
        
        # Ignore contour that are too black
        num_black_pixels = np.sum(contour < 65)
        contour_area = cv2.contourArea(contour)
        if num_black_pixels > contour_area * 0.5:
            continue
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        circularity = 0
        # Calculate circularity (perfect circle has circularity close to 1)
        if perimeter != 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)

        # Draw all contours
        cv2.drawContours(vis_contours, [contour], 0, (0, 255, 0), 2)

        # Set a threshold for circularity and minimum area
        if area >= area_min and area <= area_max and circularity >= circularity_min: 
            red_dots += 1
            
            if return_visualization:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                cv2.circle(vis_filtered, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    if tongue_percentage >= 40:
        red_dot_threshold = 20
    elif tongue_percentage >= 20:
        red_dot_threshold = 17
    else:
        red_dot_threshold = 15
        
    if return_visualization:
        vis_result = original_image.copy()
        cv2.putText(vis_result, f"Red dots: {red_dots}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        vis_result = add_label(vis_result, "Result")

        top_row = np.hstack((vis_original, vis_gray, vis_enhanced))
        middle_row = np.hstack((vis_blurred, vis_thresh, vis_contours))
        bottom_row = np.hstack((vis_filtered, vis_result, np.zeros_like(original_image)))
        
        grid_visualization = np.vstack((top_row, middle_row, bottom_row))

    # print(red_dots)
    # print(red_dot_threshold)
    print(tongue_percentage)
    
    return red_dots >= red_dot_threshold, grid_visualization


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
            
    #         num_cracks, visualization = detect_red_dots_with_visualization(image)

    #         # Save visualization
    #         vis_path = 'Test Images/Crack Visualizations/test_visualization_cracks' + image_file
    #         cv2.imwrite(vis_path, visualization)
            
    # Test single image
    image_path = 'Samples/sample_9.jpg'
    image = cv2.imread(image_path)
    result, visualization = detect_red_dots(image, return_visualization=True)
    print(result)
    cv2.imwrite('Visualizations/test_visualization_red_dots_single.jpg', visualization)
    
if __name__ == "__main__":
    main()
