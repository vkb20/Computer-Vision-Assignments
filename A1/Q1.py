from sklearn.cluster import KMeans
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from math import *
from scipy import ndimage
from filter import *
from segment_graph import *
from main import *

#if eqNo=3 then eq3 solution will run and if eqNo=5 then eq5 will run
eqNo = int(input("enter the equation number that you want to run : "))

if(eqNo==3):
	#taking image as input
	orgImg = cv2.imread("./leaf.png")
	M = len(orgImg)
	N = len(orgImg[0])

	#reshaping image to (M*N)x3.
	img = orgImg.reshape((orgImg.shape[0] * orgImg.shape[1], 3))

	#labels contains the corresponding cluster a particular index [B,G,R] value is assigned to.
	#clusters contains the 85 different [B,G,R] values.
	kmeans = KMeans(n_clusters = 85).fit(img)
	labels = kmeans.labels_
	clusters = np.array(kmeans.cluster_centers_, dtype = np.uint8)

	#countMap to store count of each cluster in the image.
	countMap = np.zeros((85, 4))

	#first three indices to store the 85 unique colors and 4th index to store the saliency value.
	saliencyMap = np.zeros((85, 4))

	#here i am assigning the 85 unique colors to first three indices of count and saliency map.
	for i in range(85):
		countMap[i][0] = clusters[i][0]
		countMap[i][1] = clusters[i][1]
		countMap[i][2] = clusters[i][2]
		saliencyMap[i][0] = clusters[i][0]
		saliencyMap[i][1] = clusters[i][1]
		saliencyMap[i][2] = clusters[i][2]

	#each index of label will have a index which corresponds to a cluster. so using that index
	#i am storing the count of a particular cluster.
	for i in range(M*N):
		idx = labels[i]
		countMap[idx][3] = countMap[idx][3]+1

	#here i looping through 85 unique values and for a particular "i" i am obtaining the saliency value.
	#saliency value is obtained by summing the of euclidean distance between the "ith"
	#color with every other color multiplied by probability of every other color.
	for i in range(85):
		b1 = saliencyMap[i][0]
		g1 = saliencyMap[i][1]
		r1 = saliencyMap[i][2]
		salSum = 0
		for j in range(85):
			b2 = saliencyMap[j][0]
			g2 = saliencyMap[j][1]
			r2 = saliencyMap[j][2]
			dist = sqrt((b1-b2)**2 + (g1-g2)**2 + (r1-r2)**2)
			prob = countMap[j][3]/(M*N)
			salSum = salSum + prob*dist
		saliencyMap[i][3] = salSum

	#here i am obtaining index value corresponding to cluster and then assigning the saliency value
	#corresponding to the cluster.
	outImg = np.zeros((M,N), dtype = np.uint8)
	c = 0
	for i in range(M):
		for j in range(N):
			idx = labels[c]
			salVal = int(saliencyMap[idx][3])
			outImg[i][j] = salVal
			c=c+1

	cv2.imwrite("eq_3_out.png", outImg)
	cv2.imshow("org", orgImg)
	cv2.imshow("out", outImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

elif(eqNo==5):
	#regions contains different regions as keys and cells that are in that region are values.
	regions = {}
	#countOfDiffColorsInRegion contains regions as key and dictionary as value.
	#the dictionary which is value has key as colors and value as count.
	countOfDiffColorsInRegion = {}
	#saliencyOfRegions contains saliency of each region. 
	saliencyOfRegions = {}

	#taking image as input
	orgImg = cv2.imread("./BigTree.jpg")
	M = len(orgImg)
	N = len(orgImg[0])

	#reshaping image to (M*N)x3.
	img = orgImg.reshape((orgImg.shape[0] * orgImg.shape[1], 3))

	#labels contains the corresponding cluster a particular index [B,G,R] value is assigned to.
	#clusters contains the 85 different [B,G,R] values.
	kmeans = KMeans(n_clusters = 85).fit(img)
	labels = kmeans.labels_
	clusters = np.array(kmeans.cluster_centers_, dtype = np.uint8)

	#here i am obtaining the image which only contains 85 colors obtained using kmeans algo.
	clusteredImage = np.zeros((M,N,3), dtype = np.uint8)
	c = 0
	for i in range(M):
		for j in range(N):
			idx = labels[c]
			clusteredImage[i][j] = clusters[idx]
			c = c+1

	sigma = 0.5
	k = 500
	minx = 50
	segmentedImage = segment(clusteredImage, sigma, k, minx)

	#here each segmentedImage cell consist of array which denotes the region it belongs to.
	#so converting that array to string and then assigning a list to each string region.
	#this list contains the cells that belong to a particular region.
	for i in range(M):
		for j in range(N):
			strKey = str(segmentedImage[i][j])
			strKey = strKey[1:len(strKey)-1]
			if(strKey in regions):
				getList = regions.get(strKey)
				getList.append([i,j])
				regions[strKey] = getList 
			else:
				regions[strKey] = [[i,j]]

	#here i am obtaining the unique colors and their count in a particular region.
	#each region is a key and has a value as dictionary and this dictionary has
	#key as unique colors as string and their count as value.
	for key in regions:
		cellsList = regions[key]
		colorsMap = {}
		for i in range(len(cellsList)):
			col = clusteredImage[cellsList[i][0]][cellsList[i][1]]
			strCol = str(col)
			strCol = strCol[1:len(strCol)-1]
			if(strCol in colorsMap):
				getCount = colorsMap.get(strCol)
				getCount = getCount+1
				colorsMap[strCol] = getCount
			else:
				colorsMap[strCol] = 1

		countOfDiffColorsInRegion[key] = colorsMap

	#here i am looping through each "ith" region. then in nested loop i am looping through
	#each "jth" region. then for "ith" and "jth" region i am looping through each color
	#of "ith" region and in nested loop i am looping through each color of "jth" region.
	#then i am determing for each colors their probability in their region and their 
	#euclidean distance. then doing these calculations for all the combinations of the "ith" and "jth"
	#region. then i am summing them up and multiplying with the weight of "jth" region and then
	#adding it to the saliency value of the "ith" region.
	maxSaliency = -1
	for key in regions:
		colorInR1 = countOfDiffColorsInRegion[key]
		totalSal = 0
		for key2 in regions:
			dSum = 0
			if(key!=key2):
				weight = len(regions[key2])
				colorInR2 = countOfDiffColorsInRegion[key2]
				for col1 in colorInR1:
					colList = col1.split(" ")
					idx = 0
					for loop in range(0,len(colList)):
						if(colList[loop]!=""):
							if(idx==0):
								b1 = int(colList[loop])
								idx=idx+1
							elif(idx==1):
								g1 = int(colList[loop])
								idx=idx+1
							elif(idx==2):
								r1 = int(colList[loop])
								break
					prob1 = colorInR1[col1]/len(regions[key])
					for col2 in colorInR2:
						colList2 = col2.split(" ")
						idx = 0
						for loop in range(0,len(colList2)):
							if(colList2[loop]!=""):
								if(idx==0):
									b2 = int(colList2[loop])
									idx=idx+1
								elif(idx==1):
									g2 = int(colList2[loop])
									idx=idx+1
								elif(idx==2):
									r2 = int(colList2[loop])
									break
						prob2 = colorInR2[col2]/len(regions[key2])
						dist = sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
						dSum = dSum+(prob1*prob2*dist)
				totalSal = totalSal + dSum*weight

		saliencyOfRegions[key] = totalSal
		maxSaliency = max(maxSaliency, totalSal)

	#here i am normalizing the saliency values from 0 to 255.
	for key in saliencyOfRegions:
		getVal = saliencyOfRegions[key]
		getVal = getVal/maxSaliency
		getVal = int(255*getVal)
		saliencyOfRegions[key] = getVal

	#here i am assigning the saliency values to the each cell of output image by using the
	#"regions" dictionary which contains the information of each cell belonging
	#to a particular region.
	outImg = np.zeros((M,N), dtype = np.uint8)
	for key in regions:
		cellsList = regions[key]
		saliencyVal = saliencyOfRegions[key]
		for i in range(len(cellsList)):
			outImg[cellsList[i][0]][cellsList[i][1]] = saliencyVal

	cv2.imwrite("eq_5_out.png", outImg)
	cv2.imshow("in", clusteredImage)
	cv2.imshow("seg", segmentedImage)
	cv2.imshow("out", outImg)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


