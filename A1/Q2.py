import cv2
from math import * 
import numpy as np
import csv

#taking image as input
img = cv2.imread("./horse.jpg")
M = len(img)
N = len(img[0])

#converting the RGB image to grayscale
grayscaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#list for storing sum of TSS for each threshold
sumOfTSSList = []

#threshold and corresponding minimum TSS sum
thresholdForMinSumTss = 0
TSS = 100000000000000000

#for determining which part to be given 255 value and which part to be given 0 value in output.
determineMask = False

#looping through 0 to 255 (for threshold)
#then dividing into two parts i.e. pixel values less than or equal to threshold and pixel values more than threshold.
#After that calculating the mean of pixel values less than or equal to threshold and mean of pixel values more than threshold.
#After that calculating total sum of squares for set of values less than or equal to threshold and for set of values more than threshold.
#After that computing the sum of TSS for both cases.
#Then determining for which threshold this sum is minimum.
for i in range(256):
	threshold = i
	sumLessOrEq = 0
	sumMore = 0
	countLessOrEq = 0
	countMore = 0
	for j in range(M):
		for k in range(N):
			if(grayscaleImg[j][k]<=threshold):
				sumLessOrEq = sumLessOrEq+grayscaleImg[j][k]
				countLessOrEq = countLessOrEq+1
			else:
				sumMore = sumMore+grayscaleImg[j][k]
				countMore = countMore+1

	meanLessOrEq = 0
	meanMore = 0
	if(countLessOrEq!=0):
		meanLessOrEq = sumLessOrEq/countLessOrEq
	if(countMore!=0):
		meanMore = sumMore/countMore
	
	TSSLessOrEq = 0
	TSSMore = 0
	for j in range(M):
		for k in range(N):
			if(grayscaleImg[j][k]<=threshold):
				TSSLessOrEq = TSSLessOrEq + ((float(grayscaleImg[j][k])-meanLessOrEq)**2)
			else:
				TSSMore = TSSMore + ((float(grayscaleImg[j][k])-meanMore)**2)

	tempSum = TSSLessOrEq+TSSMore
	if(tempSum<TSS):
		TSS = tempSum
		thresholdForMinSumTss = threshold
		if(countLessOrEq<=countMore):
			determineMask = True
		else:
			determineMask = False

	temp = [threshold, tempSum]
	sumOfTSSList.append(temp)

#writing the threshold data and its corresponding sum of TSS in the CSV file.
heading = ['threshold', 'sum of TSS']
with open("sum_of_TSS_file.csv", 'w', newline='') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(heading)
	csvwriter.writerows(sumOfTSSList)

#depending on whether the count of pixels less than or equal to threshold was high 
#or count of pixels more than threshold was high i am assigning the data to the binaryMaskImg matrix. 
#For the part which has low count is assigned 255 and 0 is assigned to other part.
binaryMaskImg = np.zeros((M, N), dtype = np.uint8)
for i in range(M):
	for j in range(N):
		if(grayscaleImg[i][j]<=thresholdForMinSumTss):
			if(determineMask):
				binaryMaskImg[i][j] = 255
			else:
				binaryMaskImg[i][j] = 0
		else:
			if(determineMask):
				binaryMaskImg[i][j] = 0
			else:
				binaryMaskImg[i][j] = 255

#saving the output image as png
cv2.imwrite("binaryMask.png", binaryMaskImg)

cv2.waitKey(0)
cv2.destroyAllWindows()


