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

def adjust_gamma(frame,gamma=1.0):
	#build a lookup table mapping the pixel values [0,255] to
	#their adjusted gamma values
	invGamma = 1.0/gamma
	table = np.array([((i/255.0)**invGamma)*255
		for i in np.arange(0,256)]).astype("uint8")
	return cv2.LUT(frame,table)


while(1):

	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	frame = imutils.resize(frame,width=320)
	
	#frame = adjust_gamma(frame,0.45)

	#scale = 0.75
	#frame = (frame*scale).astype(np.uint8)


    	frame2 = cv2.imread("stop.jpg")
  	#frame2 = cv2.resize(frame2, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
	frame2 = imutils.resize(frame2,width=320)
    	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = (gray*0.3).astype(np.uint8)	
	gray = adjust_gamma(gray,0.4)
	#gray = cv2.equalizeHist(gray)
	#gray = np.hstack((gray,equ))
		
	gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	cv2.imshow('gray',gray)
	cv2.imwrite('gray1.png',gray)
	cv2.imshow('gray2',gray2)
	cv2.imwrite('gray2.png',gray2)
	stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
	stops2 = stop_cascade.detectMultiScale(gray2, 1.4, 5)
   	#stops3 = stop_cascade.detectMultiScale(gray, 1.5, 5)
    
    
  	if len(stops) == 0:
		print "found nothing" 
	for (x,y,w,h) in stops:
		print "found somethin"
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
      	
	for (x,y,w,h) in stops2:
		print "found somethin2"
		cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,255,0),2)

	end_time = time.time()
	timee = end_time-start_time
	#print timee
	cv2.imshow('img',frame)
	cv2.imwrite('frame1.png',frame)
	cv2.imshow('img2',frame2)
	cv2.imwrite('frame2.png',frame2)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cv2.destroyAllWindows()
vs.stop()
