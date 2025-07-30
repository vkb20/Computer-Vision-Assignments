import numpy as np
import cv2
from math import *
from sklearn.cluster import DBSCAN
import os

'''
HOW TO RUN : 
Change the path where you want to write the output. change path for
input image also. Change path in lines 14 and 97.
'''

#input image
img = cv2.imread('../inputs/0002.jpg')
M = len(img)
N = len(img[0])

#converting image to (M*N)x3
rImg = img.reshape((img.shape[0] * img.shape[1], 3))

#For storing the count of clusters, outliers and corepoints.
storeClustersOutliersCore = []

#different epsilon and minimum samples.
epsilon = [1, 1, 2, 3, 5, 5, 6, 8]
minSamps = [3, 10, 30, 15, 5, 25, 33, 2]

#looping through epsilon, minSamps list.
for pp in range(len(epsilon)):

	#applying inbuilt DBSCAN and obtaining labels.
	DBSCANCluster = DBSCAN(eps=epsilon[pp], min_samples=minSamps[pp])
	DBSCANCluster.fit(rImg)
	labels = DBSCANCluster.labels_

	outliers = 0
	clusters = 0
	total = 0
	k = 0
	locationCorrespondingToLabels = {}

	#here I am obtaining all locations matrix for particular label and storing in the dictionary.
	for i in range(M):
		for j in range(N):
			label = labels[k]
			if(label!=-1):
				total = total + 1
				if(label in locationCorrespondingToLabels):
					getLocation = locationCorrespondingToLabels[label]
					getLocation.append([i,j])
					locationCorrespondingToLabels[label] = getLocation
				else:
					locationCorrespondingToLabels[label] = [[i,j]]
					clusters = clusters + 1
			else:
				outliers = outliers + 1
			k = k+1

	#here for each label obtained I am storing the mean of R,G,B values.
	#these RGB values belonging to a label will be assigned to all cells having that label.
	meanBGRForLabels = {}
	for l in locationCorrespondingToLabels:
		if(l!=-1):
			getLocation = locationCorrespondingToLabels[l]
			meanBGR = [0,0,0]
			for i in range(len(getLocation)):
				x = getLocation[i][0]
				y = getLocation[i][1]
				BCol = img[x][y][0]
				GCol = img[x][y][1]
				RCol = img[x][y][2]
				meanBGR[0] = meanBGR[0] + BCol
				meanBGR[1] = meanBGR[1] + GCol
				meanBGR[2] = meanBGR[2] + RCol

		for i in range(0,3):
			meanBGR[i] = int(meanBGR[i]/len(getLocation))

		meanBGRForLabels[l-1] = meanBGR

	#appending clusters, outliers and core points count
	storeClustersOutliersCore.append([len(locationCorrespondingToLabels), outliers, M*N - outliers - len(locationCorrespondingToLabels)])

	#Here I am just assigning the mean R,G,B obtained to each cell depending on label it belongs to.
	#For label = -1, I am keeping it as black.
	outImg = np.zeros((M,N,3), dtype = np.uint8)
	k = 0
	for i in range(M):
		for j in range(N):
			label = labels[k]
			if(label!=-1):
				outImg[i][j][0] = meanBGRForLabels[label-1][0]
				outImg[i][j][1] = meanBGRForLabels[label-1][1]
				outImg[i][j][2] = meanBGRForLabels[label-1][2]
			k = k+1

	cv2.imwrite(os.path.join("..", "outputs", f"eps_{epsilon[pp]}_samples_{minSamps[pp]}.jpg"), outImg)

print(storeClustersOutliersCore)
		
