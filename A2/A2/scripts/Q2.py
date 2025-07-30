import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
from fcmeans import FCM
import os

'''
HOW TO RUN : 
You can run this code in command prompt.
Change the path from path of input images where you've downloaded it.
Also cluster directory should be according to the path where you want to create it.
change paths in line 166,176,186,187. 
'''


'''
In this function I am passing matrix (which is 256x256 image).
Then I am looping through the matrix cells and looking for neighboring cells.
then if the neighboring indices are valid then i am calculating minimum and
maximum of (neigboring, current value) and then calculating min/max and
then for each 8 neighbors of the cell I am doing this calculation and then
converting the binary string to integer and storing this integer value in 
(i,j) cell of LBPCodeMatrix. 
'''
def modifiedLBP(matrix):
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
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i-1>=0 and j>=0):
				n = matrix[i-1][j]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i-1>=0 and j+1<len(matrix[0])):
				n = matrix[i-1][j+1]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i>=0 and j+1<len(matrix[0])):
				n = matrix[i][j+1]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i+1<len(matrix) and j+1<len(matrix[0])):
				n = matrix[i+1][j+1]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i+1<len(matrix) and j>=0):
				n = matrix[i+1][j]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i+1<len(matrix) and j-1>=0):
				n = matrix[i+1][j-1]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"

			if(i>=0 and j-1>=0):
				n = matrix[i][j-1]
				neighbor = neighbor + str(n)
				bit = 0
				if(max(n,c)!=0):
					bit = round(min(n,c)/max(n,c))
				stringLBP = stringLBP + str(bit)
			else:
				stringLBP = stringLBP + "0"
			LBPcode = int(stringLBP, 2)
			LBPCodeMatrix[i][j] = LBPcode
	return LBPCodeMatrix

'''
Here I am taking LBPCodeMatrix of a image as function argument and then I am 
defining patches of size 4x4, 8x8....., 256x256. Then for each patch size 
I am generating a histogram with 16 bins where 0th bin correspond to [0,16), 
1st bin correspond to [16,32) and so on till 15th bin. Then for each patch I am
checking the LBP value at (i,j) and then checking the corresponding bin and 
increase the count corresponding to that bin. Then for patch size 4x4, the 
total number of histograms would be (256*256)/(4*4) and similarly for other
patch size as well. Then I am basically concatenating all the histograms of 
all the patch size of the image. So for a image the number of histograms would
be 5461 as I have choosen patchSize as [4,8,16,32,64,128,256].
'''
def SPP(LBPCodeMatrix):
	p = 0
	imageLevelHistogram = np.zeros(5461*16)
	patchPow = [4,8,16,32,64,128,256]
	for t in range(len(patchPow)):
		patchSize = patchPow[t]
		for i in range(0, 256, patchSize):
			for j in range(0, 256, patchSize):
				LBPPatch = LBPCodeMatrix[i:i+patchSize, j:j+patchSize]
				histogram = np.zeros(16)
				for k in range(len(LBPPatch)):
					for l in range(len(LBPPatch[0])):
						patchVal = LBPPatch[k][l]
						idx = int(patchVal/16)
						if(idx==16):
							histogram[15] = histogram[15] + 1
						else:
							histogram[idx] = histogram[idx] + 1
				pp = 0
				temp = p
				while(temp-p<=15):
					imageLevelHistogram[p] = histogram[pp]
					pp = pp+1
					temp = temp+1
				p = temp

	return imageLevelHistogram


k_inp = int(input("enter k : "))

'''
Here I am basically looping through all the images in dataset and then obtaining
the 5461*16 histogram and then putting it in the combinedHistogram matrix.
'''
combinedHistogram = np.zeros((30, 5461*16))
for i in range(1,31):
	imgPath = f"../inputs/{i}.jpg"
	img = cv2.imread(imgPath)
	img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	LBPCodeMatrix = modifiedLBP(img)
	imageLevelHistogram = SPP(LBPCodeMatrix)
	combinedHistogram[i-1] = imageLevelHistogram

#creating "k" cluster folders.
for i in range(0, k_inp):
    os.makedirs(os.path.join("..", "outputs", "clusters", f"cluster{i+1}"))

#using fuzzy c-means to group the images into k-clusters.
fMeans = FCM(n_clusters=k_inp)
fMeans.fit(combinedHistogram)
labels = fMeans.predict(combinedHistogram)

#putting the images in the corresponding cluster
for i in range(len(labels)):
	idx = i+1
	path = os.path.join("..", "outputs", "clusters", f"cluster{int(labels[i])}")
	img = cv2.imread(os.path.join("..", "inputs", f"{idx}.jpg"))
	cv2.imwrite(os.path.join(path, f"{idx}.jpg"), img)


