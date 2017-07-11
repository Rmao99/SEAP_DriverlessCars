import numpy as np
import cv2

cap = cv2.VideoCapture(0)
stop_cascade = cv2.CascadeClassifier('cascade.xml')
while(1):

	_,frame = cap.read()
    #frame = cv2.imread("stop.jpg")
   # frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
    
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
    
   
	for (x,y,w,h) in stops:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
      
	cv2.imshow('img',frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cv2.destroyAllWindows()
