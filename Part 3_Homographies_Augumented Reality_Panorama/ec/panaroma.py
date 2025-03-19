import cv2
import numpy as np

def load_images(img1_path, img2_path):
    # Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    return img1, img2

def find_match_features(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoint1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoint2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match features using FLANN-based matcher
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Use Lowe's ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)


    # Find the locations of matched keypoints
    point1 = np.float32([keypoint1[m.queryIdx].pt for m in good_matches])
    point2 = np.float32([keypoint2[m.trainIdx].pt for m in good_matches])

    return point1, point2

def compute_homography(point1, point2):
   # Use RANSAC for homography
    homography, _ = cv2.findHomography(point2, point1, cv2.RANSAC, 5.0)
    return homography

def stitch_images(img1, img2, homography):
    height, width, _ = img1.shape
    panorama = cv2.warpPerspective(img2, homography, (width + img2.shape[1], height))
    panorama[0:height, 0:width] = img1
    return panorama

def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Save to {output_path}")

def main():
    img1_path = "../data/left.jpg"  
    img2_path = "../data/right.jpg" 

    img1, img2 = load_images(img1_path, img2_path)

    # Find and then match features
    try:
        point1, point2 = find_match_features(img1, img2)
    except RuntimeError as e:
        print(e)
        return

    # Compute homography
    homography = compute_homography(point1, point2)

    # Stitch the images
    panorama = stitch_images(img1, img2, homography)

    # Save 
    save_image(panorama, "../data/bara1.jpg")

if __name__ == "__main__":
    main()
