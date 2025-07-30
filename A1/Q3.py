import numpy as np
import cv2
from math import *

#storing each frame of video in this matrix.
storeRGBForEachFrame = []

#the video has 84 frames. The frames matrix will be store inside storeRGBForEachFrame.
vid = cv2.VideoCapture('shahar_walk.avi')
i = 0
while(vid.isOpened()):
  ret, frame = vid.read()
  if(i==84):
    break
  else:
    storeRGBForEachFrame.append(frame)
  i=i+1

L = len(storeRGBForEachFrame)
M = len(storeRGBForEachFrame[0])
N = len(storeRGBForEachFrame[0][0])

#for storing median [B,G,R] corresponding to each cell of the image
medianMatrix = np.empty((M, N, 3), dtype = np.uint8)

#here i am looping each cell (j,k) of every "ith" image.
#then storing the [B,G,R] values of (j,k) cell of every image.
#then i am sorting them based on comparing B,G,R values.
#then computing the median [B,G,R] of each (j,k) cell and storing in medianMatrix.
for j in range(M):
  for k in range(N):
    jkRGBMatrix = []
    for i in range(L):
      temp = [storeRGBForEachFrame[i][j][k][0], storeRGBForEachFrame[i][j][k][1], storeRGBForEachFrame[i][j][k][2]]
      jkRGBMatrix.append(temp)
    
    tempMedian = []
    sortedMerged = sorted(jkRGBMatrix,key = lambda x:(256*256*int(x[0])+256*int(x[1])+int(x[2])))

    if(L%2==0):
      mid = L//2
      midn1 = mid-1
      tempMedian = [(int(sortedMerged[mid][0])+int(sortedMerged[midn1][0]))/2, (int(sortedMerged[mid][1])+int(sortedMerged[midn1][1]))/2, (int(sortedMerged[mid][2])+int(sortedMerged[midn1][2]))/2]
    else:
      mid = L//2
      tempMedian = [sortedMerged[mid][0], sortedMerged[mid][1], sortedMerged[mid][2]]

    medianMatrix[j][k] = tempMedian

eachFrameBinaryMask = []

#here for each image "i" am subtracting [B,G,R] values of "ith" image matrix with medianMatrix.
#then obviously this will lead to many background cells being close to zero.
#then if subtracted R,G,B values mean is less than threshold that means it is background.
#otherwise it is foreground.
for i in range(L):
  ithFrame = np.array(storeRGBForEachFrame[i], dtype = np.uint8)
  diff = np.zeros((M,N,3), dtype = np.uint8)
  for j in range(M):
    for k in range(N):
      diff[j][k][0] = int(abs(int(ithFrame[j][k][0])-int(medianMatrix[j][k][0])))
      diff[j][k][1] = int(abs(int(ithFrame[j][k][1])-int(medianMatrix[j][k][1])))
      diff[j][k][2] = int(abs(int(ithFrame[j][k][2])-int(medianMatrix[j][k][2])))
  out = np.zeros((M, N), dtype = np.uint8)
  for j in range(M):
    for k in range(N):
      intensity = (int(diff[j][k][0])+int(diff[j][k][1])+int(diff[j][k][1]))/3
      if(intensity>40):
        out[j][k] = 255
      else:
        out[j][k] = 0
  eachFrameBinaryMask.append(out)
  cv2.waitKey(1)

#here for each frame we've a binary mask. 
#so wherever values are 1, i am calculating min and max, (x,y) coordinates.
#after that i am obtaining the center and radius through those values.
#then i am drawing the circle around the center coordinates with radius "rad".
for i in range(L):
  ithBinMask = eachFrameBinaryMask[i]
  minX = 1000000
  minY = 1000000
  maxX = -1
  maxY = -1
  for j in range(M):
    for k in range(N):
      if(ithBinMask[j][k]==255):
        minX = min(minX, k)
        minY = min(minY, j)
        maxX = max(maxX, k)
        maxY = max(maxY, j)
  if((maxX-minX)%2!=0):
    maxX=maxX+1
  if((maxY-minY)%2!=0):
    maxY=maxY+1
  centerX = (minX+maxX)//2
  centerY = (minY+maxY)//2
  rad = maxY-centerY
  storeRGBForEachFrame[i] = cv2.circle(storeRGBForEachFrame[i], (centerX, centerY), rad, (0, 0, 255), 2)
  
#here i am assigning each frame to the video with 42 fps.
out_vid = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 42, (N, M))

for i in range(L):
  out_vid.write(storeRGBForEachFrame[i])
  
cv2.destroyAllWindows()
vid.release()


