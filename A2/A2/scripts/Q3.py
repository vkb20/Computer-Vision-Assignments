import numpy as np
import cv2
import math
from sklearn.cluster import KMeans

'''
HOW TO RUN : 
You can run this code in command prompt.
Change the path from path of input images where you've downloaded it.
change path in line 91 and 113.
'''


'''
In this function, I am taking image as argument and then applying 
sobel filter to obtain the edges in "x" and "y" direction. After that
for each cell I am calculating magnitude as square root of sum of squares of the 
values present in the gradX and gradY. Also obtaining the angle at each cell.
'''
def magAndAngle(img):
	gradX = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
	gradY = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

	mag = np.zeros((len(gradX), len(gradX[0])))
	angle = np.zeros((len(gradX), len(gradX[0])))

	for i in range(len(gradX)):
		for j in range(len(gradX[0])):
			x = gradX[i][j]
			y = gradY[i][j]
			mag[i][j] = math.sqrt(x**2 + y**2)
			if(x>0):
				angle[i][j] = math.degrees(math.atan(y/x))
			elif(x==0):
				if(y>0):
					angle[i][j] = math.degrees(math.pi/2)
				elif(y<0):
					angle[i][j] = math.degrees(-math.pi/2)
				else:
					angle[i][j] = 0.0
			else:
				if(y>=0):
					angle[i][j] = math.degrees(math.atan(y/x)+math.pi)
				else:
					angle[i][j] = math.degrees(math.atan(y/x)-math.pi)

	mAndD = [mag, angle]
	return mAndD

'''
In this function I am taking 16x16 patch of magnitude as well as angle as argument.
Then for each cell in the patch I am calculating the angle range it belongs to.
If angle = 35 then it belongs to range [30,60). I have created 13 bins i.e. 
0th bin is [-180, -150), 1st bin is [-150, -120) and so on. Then if the angle is
divisible by 30 then the whole magnitude goes to only the (angle+180)/30 th bin.
Otherwise I am obtaining the left and right angles of the angle and then dividing
the magnitude according to how close the angle is to its left and right key angles.
Then I am creating a histogram of magnitude sum for each bin.
'''
def HOG(magPatch, angPatch):
	histogram = np.zeros(13)
	for i in range(len(magPatch)):
		for j in range(len(magPatch[0])):
			ang = angPatch[i][j]
			mag = magPatch[i][j]
			left = (ang+180)/30
			if(ang%30==0):
				histogram[int(left)] = histogram[int(left)] + mag
			else:
				leftAng = int(left)*30 - 180
				rightAng = leftAng + 30
				leftDiff = abs(leftAng-ang)
				rightDiff = abs(rightAng-ang)
				leftMag = mag*(1 - ((leftDiff)/(30)))
				rightMag = mag*(1 - ((rightDiff)/(30)))
				histogram[int(left)] = histogram[int(left)] + leftMag
				histogram[int(left)+1] = histogram[int(left)+1] + rightMag
	
	return histogram


k_inp = int(input("enter k : "))

'''
Here I am obtaining the histogram of each patch of each image and then adding the histogram to
patchesHisto matrix. These histograms are storing information about each patch of the image.
'''
patchesHisto = np.zeros((256*20, 13))
p = 0
for i in range(1,21):
	imgPath = f"../inputs/{i}.jpg"
	img = cv2.imread(imgPath)
	img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	magAndAng = magAndAngle(img)
	mag = magAndAng[0]
	angle = magAndAng[1]
	for j in range(0, 256, 16):
		for k in range(0, 256, 16):
			magPatch = mag[j:j+16, k:k+16]
			angPatch = angle[j:j+16, k:k+16]
			histogram = HOG(magPatch, angPatch)
			for z in range(13):
				patchesHisto[p][z] = histogram[z]
			p = p+1

#obtaining labels and clusters using k-means. This would give k-representative patches.
kmeans = KMeans(n_clusters = k_inp).fit(patchesHisto)
labels = kmeans.labels_
clusters = np.array(kmeans.cluster_centers_)

#Here I am obtaining magnitude, angle matrix of the searched image.
testImg = cv2.imread("../inputs/search.jpg")
testImg = cv2.resize(testImg, (256, 256), interpolation = cv2.INTER_CUBIC)
testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
magAndAng = magAndAngle(testImg)
mag = magAndAng[0]
angle = magAndAng[1]

'''
Here I am obtaining the feature vector of searched image. I am obtaining 
the histogram of patches of searched image and then calculating the 
euclidean distance between representative patches and searched image patches.
Then the count of representative patch with minimum distance is increased.
'''
featureVect = np.zeros(k_inp)
for j in range(0, 256, 16):
	for k in range(0, 256, 16):
		magPatch = mag[j:j+16, k:k+16]
		angPatch = angle[j:j+16, k:k+16]
		histogram = HOG(magPatch, angPatch)
		lessDistIdx = -1
		lessDist = 10000000000000
		for z in range(k_inp):
			dist = 0
			for zz in range(13):
				dist = dist + (clusters[z][zz] - histogram[zz])**2
			dist = math.sqrt(dist)
			if(dist<lessDist):
				lessDistIdx = z
				lessDist = dist
		featureVect[lessDistIdx] = featureVect[lessDistIdx] + 1

print(featureVect)

