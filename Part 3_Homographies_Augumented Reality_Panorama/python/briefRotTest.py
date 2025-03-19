import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


#Q3.5
#Read the image and convert to grayscale, if necessary, you can use OpenCV
cv_cover = cv2.imread('../data/cv_cover.jpg')
rotation_angles = np.arange(0, 360, 10)
match_counts = []

for angle in rotation_angles:
	#Rotate Image
    rotated_img = rotate(cv_cover, angle, reshape=False)
	
	#Compute features, descriptors and Match features
    matches, _, _ = matchPics(cv_cover, rotated_img)

	#Update histogram
    match_counts.append(len(matches))


# Display histogram
plt.figure(figsize=(10, 5))
plt.bar(rotation_angles, match_counts, width=8, color='b', alpha=0.7)
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.title('BRIEF Feature Matching vs. Rotation')
plt.xticks(rotation_angles, rotation_angles, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

