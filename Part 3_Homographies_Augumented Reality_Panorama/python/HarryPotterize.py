import numpy as np
import cv2
import skimage.io 
import skimage.color
import skimage.transform
from matchPics import matchPics
from matchPics1 import matchPics1
from planarH import computeH_ransac, compositeH
import matplotlib.pyplot as plt


#Write script for Q3.9

def main(method="BRIEF"):
    # Read images
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    
    if cv_cover is None or cv_desk is None or hp_cover is None:
        print("Could not read images.")
        return
    
    print(f"Using {method} for feature matching...")

    # Match features based on the chosen method
    if method.upper() == "BRIEF":
        matches, locs1, locs2 = matchPics(cv_desk, cv_cover)
        x1 = np.array([locs1[m[0]] for m in matches])  
        x2 = np.array([locs2[m[1]] for m in matches])  
    else:  # Use SIFT
        matches, keypoints1, keypoints2 = matchPics1(cv_desk, cv_cover)
        x1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
        x2 = np.array([keypoints2[m.trainIdx].pt for m in matches]) 

    print(f"Found {len(matches)} matching features")

    if len(matches) == 0:
        print("No matching features found.")
        return

    # Compute homography using RANSAC
    print("Computing homography using RANSAC...")
    H2to1, inliers = computeH_ransac(x1, x2)
    
    print(f"Homography computation complete. Number of inliers: {np.sum(inliers)}")

    # Resize hp_cover to match cv_cover dimensions
    print("Resizing hp_cover to match cv_cover dimensions...")
    hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

    # Create composite image
    print("Creating composite image...")
    composite_img = compositeH(H2to1, hp_cover_resized, cv_desk)

    # Print out the Composite Imge
    # Convert BGR to RGB for correct display
    composite_img_rgb = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)

    # Display using Matplotlib
    plt.imshow(composite_img_rgb)
    plt.axis("off") 
    plt.title("Harry Potterized Image")
    plt.show()

    # Save result
    output_file = 'harrypotterized11.jpg'
    cv2.imwrite(output_file, composite_img)

    print(f"Done! Composite image saved as '{output_file}'")

if __name__ == '__main__':
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "BRIEF" 
    main(method)

