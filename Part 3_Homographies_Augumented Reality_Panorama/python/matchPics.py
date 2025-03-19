import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
# #Complete functions above this line before this step
def matchPics(I1, I2):
	#I1, I2 : Images to match
	#Convert Images to GrayScale
    gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	
	#Detect Features in Both Images
    sigma = 0.15
    locs1 = corner_detection(gray1, sigma)
    locs2 = corner_detection(gray2, sigma)
	
	
	#Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(gray1, locs1)
    desc2, locs2 = computeBrief(gray2, locs2)
	

	#Match features using the descriptors
    ratio = 0.65
    matches = briefMatch(desc1, desc2, ratio)

    return matches, locs1, locs2
