import cv2
import numpy as np
import time
import imutils

from imutils.video import WebcamVideoStream

#PI VIDEO STREAM
#from imutils.video.pivideostream import PiVideoStream
#from imutils.video import FPS
#from picamera.array import PiRGBArray
#from picamera import PiCamera


#vs = PiVideoStream().start()
print "1"
#vs = WebcamVideoStream(0).start()
cap = cv2.VideoCapture(0)
print "2"
time.sleep(2.0)
oneway_cascade = cv2.CascadeClassifier('cascade.xml')
oneway_cascade2 = cv2.CascadeClassifier('onewaysignV2.xml')
print "3"
while(1):
	print "4"
	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	#frame = vs.read()
	print "5"
	_,frame = cap.read()
	print "6"
	frame = imutils.resize(frame,width=400)
    #frame = cv2.imread("stop.jpg")
   # frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
    
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ones = oneway_cascade.detectMultiScale(gray, 1.3, 5)
	ones2 = oneway_cascade2.detectMultiScale(gray, 1.3, 5)
   	#stops3 = stop_cascade.detectMultiScale(gray, 1.5, 5)
    
    
   	img = frame.copy()
	for (x,y,w,h) in ones:
		print "found somethin"
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
	for (x,y,w,h) in ones2:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	end_time = time.time()
	timee = end_time-start_time
	print timee
	cv2.imshow('img',frame)
	cv2.imshow('img2',img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cv2.destroyAllWindows()
vs.stop()
