import numpy as np
import cv2

def add_label(img, label):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h-30), (w, h), (0, 0, 0), -1)
    cv2.putText(img, label, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def get_tongue_area_percentage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the percentage of non-black pixels (tongue area)
    total_pixels = gray.shape[0] * gray.shape[1]
    tongue_pixels = np.sum(gray > 0)
    tongue_percentage = (tongue_pixels / total_pixels) * 100
    
    return tongue_percentage