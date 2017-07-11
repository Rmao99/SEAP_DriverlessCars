import cv2
import numpy as np
import time
import imutils

#PI VIDEO STREAM
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera


vs = PiVideoStream().start()
time.sleep(2.0)
stop_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')
while(1):

	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	frame = imutils.resize(frame,width=400)
    #frame = cv2.imread("stop.jpg")
   # frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
    
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
    
   
	for (x,y,w,h) in stops:
		print "found somethin"
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
      
	end_time = time.time()
	timee = end_time-start_time
	print timee
	cv2.imshow('img',frame)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cv2.destroyAllWindows()
vs.stop()
