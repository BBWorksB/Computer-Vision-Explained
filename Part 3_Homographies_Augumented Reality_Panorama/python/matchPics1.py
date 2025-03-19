import numpy as np
import cv2

def matchPics1(I1, I2):
    # Convert Images to GrayScale
    gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT Detector
    sift = cv2.SIFT_create()
    
    # Detect Keypoints and Compute Descriptors
    keypoints1, desc1 = sift.detectAndCompute(gray1, None)
    keypoints2, desc2 = sift.detectAndCompute(gray2, None)
    
    # Use BFMatcher with default parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    ratio = 0.65
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    return good_matches, keypoints1, keypoints2