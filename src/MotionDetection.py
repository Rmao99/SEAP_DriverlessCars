import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

cap = cv2.VideoCapture(0)

avg = None
motionCounter = 0
while(1):

	ret, frame = cap.read()

	if not ret:
		break

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(5,5),0)

	if avg is None:
		avg = gray.copy().astype("float")
		continue #if avg is not initialized, initialize it

	cv2.accumulateWeighted(gray,avg,0.55)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
	#select second element in output of threshold
	thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]
	
	thresh = cv2.dilate(thresh,None,iterations=2) #dilate to increase white space
	contours, heirarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for c in contours:
		if cv2.contourArea(c) < 100:
			continue

		(x,y,w,h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

	cv2.imshow("Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Delta", frameDelta)

	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()

	
