import cv2
import numpy as np

# Load the fingerprint image
image = cv2.imread('fingerprint.jpg', 0)  # Assuming the input image is named 'fingerprint.jpg'


# Apply Gaussian blur to smooth the image
blurred = cv2.GaussianBlur(image, (13,13), 0)

# Apply adaptive thresholding to segment the fingerprint ridges
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Optionally, perform morphological operations to further clean up the segmented image
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Closing operation to fill gaps

# Apply skeletalization
def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

skeleton = skeletonize(binary)

# Find minutiae points
def find_bifurcations(image):
    minutiae_points1 = []
    rows, cols = image.shape

    for r in range(5, rows - 5):
        for c in range(5, cols - 5):
            if image[r,c]==255:
                sum1 = np.sum(image[r-5:r-1, c-5:c-1], dtype=np.uint64)
                sum2 = np.sum(image[r-5:r-1, c+1:c+5], dtype=np.uint64)
                sum3 = np.sum(image[r+1:r+5, c-5:c-1], dtype=np.uint64)
                sum4 = np.sum(image[r+1:r+5, c+1:c+5], dtype=np.uint64)
                if sum1==0 or sum2==0 or sum3==0 or sum4==0:
                    if(sum1>=765 and sum2>=765 and sum3>=765):
                        minutiae_points1.append((c,r))
                        break
                    if(sum1>=765 and sum2>=765 and sum4>=765):
                        minutiae_points1.append((c,r))
                        break
                    if(sum2>=765 and sum3>=765 and sum4>=765):
                        minutiae_points1.append((c,r))
                        break
                    if(sum3>=765 and sum4>=765 and sum1>=765):
                        minutiae_points1.append((c,r))
                        break

    return minutiae_points1

def find_ridge_endings(image):
    minutiae_points2 = []
    rows, cols = image.shape

    for r in range(5,rows - 5):
        for c in range(5, cols -5):
            if image[r,c]==255:
                sum1 = np.sum(image[r-5:r-1, c-5:c-1], dtype=np.uint64)
                sum2 = np.sum(image[r-5:r+1, c+1:c+5], dtype=np.uint64)
                sum3 = np.sum(image[r+1:r+5, c-5:c-1], dtype=np.uint64)
                sum4 = np.sum(image[r+1:r+5, c+1:c+5], dtype=np.uint64)
                if sum1>=510 or sum2>=510 or sum3>=510 or sum4>=510:
                    if(sum1==0 and sum2==0 and sum3==0):
                        minutiae_points2.append((c,r))
                        r+6
                        c+6
                        continue
                    if(sum1==0 and sum2==0 and sum4==0):
                        minutiae_points2.append((c,r))
                        r+6
                        c+6
                        continue
                    if(sum2==0 and sum3==0 and sum4==0):
                        minutiae_points2.append((c,r))
                        r+6
                        c+6
                        continue
                    if(sum3==0 and sum4==0 and sum1==0):
                        minutiae_points2.append((c,r))
                        r+6
                        c+6
                        continue

    return minutiae_points2

def find_sweat_pores(image):
    minutiae_points3 = []
    rows, cols = image.shape

    for r in range(3, rows - 3):
        for c in range(3, cols - 3):
            if image[r,c]==255:
                sum1 = np.sum(image[r-3:r+3, c-3:c+3], dtype=np.uint64)
                sum2 = np.sum(image[r-2:r+2, c-2:c+2], dtype=np.uint64)
                sum3=np.sum(image[r-1:r+1,c-1:c+1],dtype=np.uint64)

                if abs(sum1-sum2)<=255 and abs(sum2-sum3)<=255: 
                    minutiae_points3.append((c,r))
                    r+4
                    c+4
                    continue

    return minutiae_points3



minutiae_points1 = find_bifurcations(skeleton)
minutiae_points2 = find_ridge_endings(skeleton)
minutiae_points3 = find_sweat_pores(skeleton)


# Mark minutiae points on the skeletalized image
skeleton_rgb1 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
skeleton_rgb2 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
skeleton_rgb3 = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)


for point in minutiae_points1:
    cv2.circle(skeleton_rgb1, point, 3, (0, 0, 255), -1)  # Larger circles with thicker outline

for point in minutiae_points2:
    cv2.circle(skeleton_rgb2, point,1, (255, 0, 0), -1)  # Larger circles with thicker outline

for point in minutiae_points3:
    cv2.circle(skeleton_rgb3, point, 2, (0, 255, 0), -1)  # Larger circles with thicker outline

print("THE MINUTIAE POINTS ARE PLOTTED IN THE IMAGE SUCCESSFULLY")


# Display the original, segmented, skeletalized images, and minutiae points
cv2.imshow('Skeletalized', skeleton)
cv2.imshow('Bifuraction', skeleton_rgb1)
cv2.imshow('Ridge Endings', skeleton_rgb2)
cv2.imshow('Sweat Pores', skeleton_rgb3)
cv2.waitKey(0)
cv2.destroyAllWindows()
