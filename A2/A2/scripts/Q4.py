import numpy as np
import cv2
import math
import os

'''
HOW TO RUN : 
You can run this code in command prompt.
Change the path from path of input images where you've downloaded it.
change path in line 121, 135, 171, 216, 217.
'''

'''
In this function I am passing matrix (which is 256x256 image).
Then I am looping through the matrix cells and looking for neighboring cells.
then if the neighboring indices are valid then i checking if n>c then I am
assigning the bit as "1" otherwise assigning it to "0".
then for each 8 neighbors of the cell I am doing this calculation and then
converting the binary string to integer and storing this integer value in 
(i,j) cell of LBPCodeMatrix. 
'''
def LBP(matrix):
	#later check if dtype = np.uint8 required
	LBPCodeMatrix = np.zeros((256, 256))
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			stringLBP = ""
			c = matrix[i][j]
			neighbor = ""
			if(i-1>=0 and j-1>=0):
				n = matrix[i-1][j-1]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i-1>=0 and j>=0):
				n = matrix[i-1][j]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i-1>=0 and j+1<len(matrix[0])):
				n = matrix[i-1][j+1]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i>=0 and j+1<len(matrix[0])):
				n = matrix[i][j+1]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i+1<len(matrix) and j+1<len(matrix[0])):
				n = matrix[i+1][j+1]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i+1<len(matrix) and j>=0):
				n = matrix[i+1][j]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i+1<len(matrix) and j-1>=0):
				n = matrix[i+1][j-1]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i>=0 and j-1>=0):
				n = matrix[i][j-1]
				neighbor = neighbor + str(n)
				bit = 0
				if(n>c):
					bit = 1
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			LBPcode = int(stringLBP, 2)
			LBPCodeMatrix[i][j] = LBPcode
	return LBPCodeMatrix

k_inp = int(input("enter k : "))

#Here in the dataset I am trying to obtain minimum corners in the whole dataset to maintain uniformity.
#Also calculating the LBP for the whole image.
LBPAllImages = np.zeros((50,256,256))
minCorners = 101
for i in range(1, 51):
	imgPath = f"../inputs/{i}.jpg"
	img = cv2.imread(imgPath)
	img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(img,100,0.1,10)
	if(len(corners)<minCorners):
		minCorners = len(corners)
	LBPAllImages[i-1] = LBP(img)

allImagesCorners = np.zeros((50, minCorners, 2))

#Here I am calculating only the strongest "minCorners" in the image.
#storing the indices of each image minCorners in "allImagesCorners".
for i in range(1, 51):
	imgPath = f"../inputs/{i}.jpg"
	img = cv2.imread(imgPath)
	img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(img,minCorners,0.1,10)
	cornersIdx = np.zeros((minCorners, 2))
	for j in range(len(corners)):
		cornersIdx[j][0], cornersIdx[j][1] = corners[j].ravel()
	allImagesCorners[i-1] = cornersIdx

keypointNeighborLBP = np.zeros((50, minCorners*16))

'''
Here for each image i am obtaining the histogram around the corner points.
For each corner the histogram is concatenated. 
'''
for i in range(len(allImagesCorners)):
	histogram = np.zeros((minCorners, 16))
	for j in range(minCorners):
		x = int(allImagesCorners[i][j][0])
		y = int(allImagesCorners[i][j][1])
		for l in range(x-2, x+3):
			for m in range(y-2, y+3):
				if(l>=0 and m>=0 and l<256 and m<256):
					keyHisto = np.zeros(16)
					LBPval = LBPAllImages[i][l][m]
					idx = int(LBPval/16)
					if(idx==16):
						keyHisto[15] = keyHisto[15] + 1
					else:
						keyHisto[idx] = keyHisto[idx] + 1
		histogram[j] = keyHisto
	histogramFlat = histogram.flatten()
	keypointNeighborLBP[i] = histogramFlat

#Here I am obtaining LBP and corners for the search image.
searchImg = cv2.imread("../inputs/search.jpg")
searchImg = cv2.resize(searchImg, (256, 256), interpolation = cv2.INTER_CUBIC)
searchImg = cv2.cvtColor(searchImg, cv2.COLOR_BGR2GRAY)
searchImgLBP = LBP(searchImg)
searchImgCorners = cv2.goodFeaturesToTrack(searchImg,minCorners,0.1,10)
searchImgHisto = np.zeros((minCorners, 16))

#Here I am obtaining histogram around the corners points of the search image.
for i in range(minCorners):
	x, y = searchImgCorners[i].ravel()
	x, y = int(x), int(y)
	for l in range(x-2, x+3):
		for m in range(y-2, y+3):
			keyHisto = np.zeros(16)
			LBPval = searchImgLBP[l][m]
			idx = int(LBPval/16)
			if(idx==16):
				keyHisto[15] = keyHisto[15] + 1
			else:
				keyHisto[idx] = keyHisto[idx] + 1
	searchImgHisto[i] = keyHisto

searchImgHistoFlat = searchImgHisto.flatten()

'''
Here I am calculating the euclidean distance between the search image histogram
and the dataset images histogram.
'''
euclideanDist = np.zeros((50, 2))
for i in range(50):
	dist = 0
	for j in range(minCorners*16):
		dist = dist + (keypointNeighborLBP[i][j]-searchImgHistoFlat[j])**2
	dist = math.sqrt(dist)
	euclideanDist[i][0] = i+1
	euclideanDist[i][1] = dist

#here I am finding the minimum k euclidean distances obtained above.
for i in range(k_inp):
	minDist = 100000000000000
	minDistIdx = -1
	for j in range(len(euclideanDist)):
		if(euclideanDist[j][1]<minDist):
			minDistIdx = j
			minDist = euclideanDist[j][1]
	minDistImg = cv2.imread(os.path.join("..", "inputs", f"{int(euclideanDist[int(minDistIdx)][0])}.jpg"))
	cv2.imwrite(os.path.join("..", "outputs", f"nearest_{i}.jpg"), minDistImg)
	euclideanDist[int(minDistIdx)][1] = 100000000000000

