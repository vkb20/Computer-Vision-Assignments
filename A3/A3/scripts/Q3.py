import time
import cv2

#input image
img = cv2.imread("../inputs/0002.jpg")

#start time for SIFT
start = time.time()

SIFT = cv2.xfeatures2d.SIFT_create()
keyP1, desc1 = SIFT.detectAndCompute(img,None)

#finish time for SIFT and start time for SURF
finish1 = time.time()

SURF = cv2.xfeatures2d.SURF_create(267)
keyP2, desc2 = SURF.detectAndCompute(img,None)
#finish time for SURF
finish2 = time.time()

print(finish1-time)
print(finish2-finish1)
