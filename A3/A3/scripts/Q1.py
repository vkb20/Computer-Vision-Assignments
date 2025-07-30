import numpy as np
import cv2
from skimage.segmentation import slic
from math import *
import os

'''
Change the path where you want to write the output. change path for
input image also. Change path in lines 13 and 113.
'''

#input image
img = cv2.imread("../inputs/0002.jpg")
M = len(img)
N = len(img[0])


#using inbuilt SLIC to obtain labels.
imgSlic = slic(img,n_segments=58)

#obtaining the maxLabel assigned to determine the map length.
maxLabel = -1
for i in range(M):
	for j in range(N):
		maxLabel = max(maxLabel, imgSlic[i][j])

#Here I am storing the colors belonging to "ith" label.
colorSegment = []

#Here I am storing the location belonging to "ith" label.
locationSegment = []

for i in range(maxLabel):
	colorSegment.append([])
	locationSegment.append([])

#creating map where I will store the saliency values corresponding to super-pixel.
saliencyMap = np.zeros(maxLabel)

#here I am appending the BGR list to the label the pixel belongs to and also appending the (X,Y) location.
for i in range(M):
	for j in range(N):
		label = imgSlic[i][j]
		colorLabel = [img[i][j][0], img[i][j][1], img[i][j][2]]
		colorSegment[label-1].append(colorLabel)
		locationSegment[label-1].append([i,j])

#here I am storing the mean of each super pixel location i.e. X,Y.
superPixelLocation = np.zeros((maxLabel, 2))

#here I am storing the mean of each super pixel color i.e. B,G,R.
superPixelColor = np.zeros((maxLabel, 3))

#after getting each color and pixel corresponding to each label,
#I am basically obtaining the mean of B,G,R for each label.
#Similarly obtaining mean of X,Y location for each label.
for i in range(maxLabel):
	ithLabelColorMatrix = colorSegment[i]
	eachColorSum = [0,0,0]
	for j in range(len(ithLabelColorMatrix)):
		eachColorSum[0] = eachColorSum[0] + ithLabelColorMatrix[j][0]
		eachColorSum[1] = eachColorSum[1] + ithLabelColorMatrix[j][1]
		eachColorSum[2] = eachColorSum[2] + ithLabelColorMatrix[j][2]
	for j in range(0,3):
		eachColorSum[j] = eachColorSum[j]/len(ithLabelColorMatrix)
	for j in range(0,3):
		superPixelColor[i][j] = eachColorSum[j]

	ithLabelLocationMatrix = locationSegment[i]
	locationSum = [0,0]
	for j in range(len(ithLabelLocationMatrix)):
		locationSum[0] = locationSum[0] + ithLabelLocationMatrix[j][0]
		locationSum[1] = locationSum[1] + ithLabelLocationMatrix[j][1]
	for j in range(0,2):
		locationSum[j] = locationSum[j]/len(ithLabelLocationMatrix)
	for j in range(0,2):
		superPixelLocation[i][j] = locationSum[j]

#I have already obtained mean for location and color for each label.
#So here I am just looping through each "ith" super-pixel and then
#through "jth" pixel and obtaining the saliency value.
maxSaliencyVal = -1
for i in range(maxLabel):
	saliencyVal = 0
	for j in range(maxLabel):
		if(i!=j):
			euclidB = (superPixelColor[i][0] - superPixelColor[j][0])**2
			euclidG = (superPixelColor[i][1] - superPixelColor[j][1])**2
			euclidR = (superPixelColor[i][2] - superPixelColor[j][2])**2
			euclidColor = sqrt(euclidB+euclidG+euclidR)
			euclidX = (superPixelLocation[i][0] - superPixelLocation[j][0])**2
			euclidY = (superPixelLocation[i][1] - superPixelLocation[j][1])**2
			euclidDist = sqrt(euclidX+euclidY)
			div = euclidDist/(sqrt((M**2)+(N**2)))
			saliencyVal = saliencyVal + (euclidColor)*(e**(-div))
	saliencyMap[i] = saliencyVal
	maxSaliencyVal = max(maxSaliencyVal, saliencyVal)

#normalizing values to 0 to 255.
for i in range(maxLabel):
	val = (saliencyMap[i]/maxSaliencyVal)*255
	saliencyMap[i] = val

#assigning saliency value obtained for each label at each cell.
outImg = np.zeros((M,N), dtype = np.uint8)
for i in range(M):
	for j in range(N):
		label = imgSlic[i][j]
		outImg[i][j] = int(saliencyMap[label-1])

cv2.imwrite(os.path.join("..", "outputs", "ans1_out.jpg"), outImg)
