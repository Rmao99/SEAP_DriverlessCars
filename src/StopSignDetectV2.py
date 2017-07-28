import numpy as np
import cv2
import time
import imutils

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera


#cap = cv2.VideoCapture(0)

#cap - cv2.VideoCapture(0)
trueWidth = 3.0 #inches
trueHeight = 3.0
focalLength = 305.5 #need to measure

vs = PiVideoStream().start()
time.sleep(2.0)
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
'''
def angle(p1,p2, p0) 
    double dx1 = p1.x - p0.x;
    double dy1 = p1.y - p0.y;
    double dx2 = p2.x - p0.x;
    double dy2 = p2.y - p0.y;
    return atan(dy1/dx1)-atan(dy2/dx2); //in rad
}'''
while(1):

   	start_time = time.time()
	#ret, frame = cap.read()
	#frame = cv2.imread("stop.jpg")
	
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	#frame = imutils.resize(frame,width=400)	

	#frame = cv2.imread("lane1_alt2")
	frame = imutils.resize(frame,width=320)	
    	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,235])

	mask = cv2.inRange(hsv,lower_red, upper_red)

	# res = cv2.bitwise_and(frame,frame, mask=mask)

	contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, contours, -1, (255,0,0),3)
			
	for cont in contours:
		area = cv2.contourArea(cont)
		if area > 150:
			print "passed check, contour area is:",area
			cnt = cont
			#perimeter = cv2.arcLength(cnt,True)
			epsilon = 0.01*cv2.arcLength(cnt,True)
			approx = cv2.approxPolyDP(cnt,epsilon,True)
		
			if len(approx) == 8:
				print "found stop sign"
				cv2.drawContours(frame,approx,-1,(0,255,255),3)
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
    
	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	# cv2.imshow('image',img)
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
