#opencv 3.2.0, python 2.7.4
import numpy as np
import cv2

#cap - cv2.VideoCapture(0)
trueWidth = 0.73
trueHeight = 0.73
focalLength = 1#need to measure
def getWidth(contours):

	max = contours[0][0][0]
	min = 10000
	
	for cont in contours:
		x1 = cont[0][0]
		if min > x1:
			min = x1
		if max < x1:
			max = x1
	return abs(max - min) 	

def getHeight(contours):

	max = contours[0][0][1]
	min = 10000

	for cont in contours:
		y1 = cont[0][1]
		if min > y1:
			min = y1
		if max < y1:
			max = y1
	return abs(max-min)

def calcDistance(width):
	return trueWidth * focalLength/ width

while(1):

	#ret, frame = cap.read()
	frame = cv2.imread("stop.jpg")
	#frame = cv2.resize(frame,None, fx=3.0,fy=3.0, interpolation = cv2.INTER_CUBIC)

	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_red = np.array([0,50,50])
	upper_red = np.array([9,255,255])

	mask = cv2.inRange(hsv,lower_red, upper_red)
	#res = cv2.bitwise_and(frame,frame,mask=mask)

	_,contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(frame,contours,-1,(255,0,0),3)

	for cont in contours:
		area = cv2.contourArea(cont)
		if area > 150:
			print "passed check, contour area is:",area
			cnt = cont
			#perimeter = cv2.arcLength(cnt,True)
			epsilon = 0.01*cv2.arcLength(cnt,True)
			approx = cv2.approxPolyDP(cnt,epsilon,True)
		
			if len(approx) ==8:
				print "found stop sign"
				cv2.drawContours(frame,cnt,-1,(0,255,255),3)
				width = getWidth(cnt)
				height = getHeight(cnt)
				
				print "width", width
				print "height", height
				
				dist = calcDistance(width)
				if dist > 10:
					print "too far"
				else:
					print "stop"
				break
				
	cv2.imshow('frame', frame)
	cv2.imshow('mask',mask)
	
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
