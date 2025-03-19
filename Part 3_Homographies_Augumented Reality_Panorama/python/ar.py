import numpy as np
import cv2
import skimage.io
import skimage.color
import skimage.transform
from planarH import computeH_ransac, compositeH

def loadVid(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def matchPics_ORB(img1, img2):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    x1 = np.array([kp1[m.queryIdx].pt for m in matches])
    x2 = np.array([kp2[m.trainIdx].pt for m in matches])
    
    return matches, x1, x2

def main():
    book_frames = loadVid('../data/book.mov')
    ar_frames = loadVid('../data/ar_source.mov')
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    
    if cv_cover is None or len(book_frames) == 0 or len(ar_frames) == 0:
        print("Error: Could not load images or videos.")
        return
    
    output_file = 'result/ar.avi'
    frame_size = (book_frames[0].shape[1], book_frames[0].shape[0])
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 30, frame_size)
    
    for i, book_frame in enumerate(book_frames):
        if i >= len(ar_frames):
            break
        
        print(f"Processing frame {i+1}/{len(book_frames)}")
        ar_frame = ar_frames[i]
        matches, x1, x2 = matchPics_ORB(book_frame, cv_cover)
        
        if len(matches) < 4:
            print("Not enough matches found, skipping frame.")
            continue
        
        H2to1, inliers = computeH_ransac(x1, x2)
        
        ar_resized = cv2.resize(ar_frame, (cv_cover.shape[1], cv_cover.shape[0]))
        
        composite_img = compositeH(H2to1, ar_resized, book_frame)
        
        out.write(composite_img)
    
    out.release()
    print(f"AR video saved as {output_file}")

if __name__ == '__main__':
    main()
