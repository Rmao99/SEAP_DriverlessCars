from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from LaneDetector import *
from PIDController import *
from State import *
from gopigo import *


import time
import cv2
import numpy as np
import imutils
import sys

#cap - cv2.VideoCapture(0)
trueWidth = 3.0 #inches
trueHeight = 3.0
focalLength = 305.5 #need to measure

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


stop_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')
vs = PiVideoStream().start()
time.sleep(2.0)
while(1):

	start_time = time.time()
	#ret, frame = cap.read()
	frame = cv2.imread("stop.jpg")
	
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	#frame = imutils.resize(frame,width=400)	

	#frame = cv2.imread("lane1_alt2")
	frame = imutils.resize(frame,width=320)	

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	stops = stop_cascade.detectMultiScale(gray, 1.3, 5)

	if len(stops) > 0:
		print "HA"
	elif len(stops) == 0:
		print "LOL"
	else:
		print "WTF"

	for (x,y,w,h) in stops:
		print "found somethin"
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		width = w
		height = h 

		dist = calcDistance(width)

		print dist
      


	'''hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_red = np.array([0,50,50])
	upper_red = np.array([9,255,255])

	mask = cv2.inRange(hsv,lower_red, upper_red)
	#res = cv2.bitwise_and(frame,frame,mask=mask)

	_,contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame,contours,-1,(255,0,0),3)

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
	'''			
	cv2.imshow('frame', frame)
	#cv2.imshow('mask',mask)
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
