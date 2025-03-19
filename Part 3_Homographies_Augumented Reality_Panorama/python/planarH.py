import numpy as np
import cv2
#Import necessary functions only
from helper import corner_detection, computeBrief, briefMatch

def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of point
    
    # Number of point pairs
    n = x1.shape[0]
    
    # Matrix A for homography computation
    A = np.zeros((2*n, 9))
    
    for i in range(n):
        x, y = x1[i]
        u, v = x2[i]
        
        # First equation for each point
        A[2*i] = np.array([-u, -v, -1, 0, 0, 0, x*u, x*v, x])
        
        # Second equation for each point
        A[2*i+1] = np.array([0, 0, 0, -u, -v, -1, y*u, y*v, y])
    
    # Eigenvector corresponding to smallest eigenvalue
    _, _, V = np.linalg.svd(A)
    
    # Last row of V has eigenvector corresponding to smallest eigenvalue
    h = V[-1]
    
    # Reshape to 3x3 homography matrix
    H2to1 = h.reshape(3, 3)
    
    return H2to1


def computeH_norm(x1, x2):
    #Q3.7
    #Compute the centroid of the points
    
    # Add homogeneous coordinate (convert to Nx3 matrices)
    x1_homogeneous = np.column_stack((x1, np.ones(x1.shape[0])))
    x2_homogeneous = np.column_stack((x2, np.ones(x2.shape[0])))
    
    # Compute centroids
    centroid1 = np.mean(x1, axis=0)
    centroid2 = np.mean(x2, axis=0)
    
    # Shift the origin of the points to the centroid
    x1_centered = x1 - centroid1
    x2_centered = x2 - centroid2
    
    # Compute scale to make largest distance sqrt(2)
    scale1 = np.sqrt(2) / np.max(np.sqrt(np.sum(x1_centered**2, axis=1)))
    scale2 = np.sqrt(2) / np.max(np.sqrt(np.sum(x2_centered**2, axis=1)))
    
    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_normalized = x1_centered * scale1
    x2_normalized = x2_centered * scale2
    
    # Similarity transform matrices
    #Similarity transform 1
    T1 = np.array([
        [scale1, 0, -scale1 * centroid1[0]],
        [0, scale1, -scale1 * centroid1[1]],
        [0, 0, 1]
    ])
    
    #Similarity transform 2
    T2 = np.array([
        [scale2, 0, -scale2 * centroid2[0]],
        [0, scale2, -scale2 * centroid2[1]],
        [0, 0, 1]
    ])
    
    # Compute homography on normalized points
    H_normalized = computeH(x1_normalized, x2_normalized)
    
    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H_normalized @ T2
    
    # Normalize to ensure H[2,2] = 1
    H2to1 = H2to1 / H2to1[2, 2]
    
    return H2to1


def computeH_ransac(x1, x2, num_iterations=1000, threshold=5):
    #Q3.8
    #Compute the best fitting homography given a list of matching points
    
    # Number of point pairs
    n = x1.shape[0]
    
    # Check if we have enough points for RANSAC min 4 
    if n < 4:
        print(f"Matching points not enough")
        # Return identity homography and empty inlier mask
        return np.eye(3), np.zeros(n, dtype=int)
    
    # Variables to track best homography
    max_inliers = 0
    bestH2to1 = None
    best_inlier_mask = np.zeros(n, dtype=bool)
    
    for i in range(num_iterations):
        random_indices = np.random.choice(n, 4, replace=False)
        
        # Get the selected points
        x1_subset = x1[random_indices]
        x2_subset = x2[random_indices]
        
        try:
            # Compute homography
            H = computeH_norm(x1_subset, x2_subset)
            
            # Convert points to homogeneous coordinates
            x2_homogeneous = np.column_stack((x2, np.ones(n)))
            
            # Transform points using homography
            x1_projected_homogeneous = H @ x2_homogeneous.T
            
            # Convert back from homogeneous coordinates
            x1_projected = x1_projected_homogeneous[:2, :] / x1_projected_homogeneous[2, :]
            x1_projected = x1_projected.T
            
            # Calculate distances between original and projected points
            distances = np.sqrt(np.sum((x1 - x1_projected)**2, axis=1))
            
            # Count points where distance is less than threshold
            inlier_mask = distances < threshold
            num_inliers = np.sum(inlier_mask)
            
            # Update best homography 
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                bestH2to1 = H
                best_inlier_mask = inlier_mask
        except:
            continue
    
    # If no  homography
    if bestH2to1 is None:
        print("No homography found.")
        bestH2to1 = np.eye(3)
    
    # Return best homography and inlier mask
    return bestH2to1, best_inlier_mask.astype(int)



def compositeH(H2to1, template, image):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography
    # Get dimensions of the destination image
    h_img, w_img = image.shape[:2]

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
    # Warp the template to align with the image using the homography matrix
    warped_template = cv2.warpPerspective(template, H2to1, (w_img, h_img))
	

	#Create mask of same size as template
    # Create a binary mask where the warped template is nonzero
    mask = np.zeros_like(warped_template, dtype=np.uint8)
    mask[warped_template > 0] = 255 
    # Convert mask to grayscale for masking
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
    # Invert the mask: White (255) for the template area, Black (0) for the background
    mask_inv = cv2.bitwise_not(mask_gray)

    # Extract the background from the original image using the inverted mask
    image_background = cv2.bitwise_and(image, image, mask=mask_inv)

    # Extract the warped template region using the original mask
    template_foreground = cv2.bitwise_and(warped_template, warped_template, mask=mask_gray)

    # Blend the two images
    composite_img = cv2.add(image_background, template_foreground)

    return composite_img
