from __future__ import division
import cv2
import numpy as np
import time
import imutils

#PI VIDEO STREAM
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera

from LaneDetector import LaneDetector


#cap = cv2.VideoCapture(0)

#THREADED PI CAM	
vs = PiVideoStream().start()
time.sleep(2.0)

#camera = PiCamera()
#camera.resolution = (640,480)
#camera.framerate = 30
#rawCapture = PiRGBArray(camera, size =(320, 240))

time.sleep(0.5)
detector = LaneDetector()

while(1):
#for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

	
	'''_, frame = cap.read()'''
	#frame = cv2.imread('lane1.jpg')
	#frame = cv2.resize(frame,(360,240))
	#frame = image.array
	#rawCapture.truncate(0)

	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	frame = imutils.resize(frame,width=400)	

	detector.process(frame)

	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
	
	print "left slope", left_slope
	print "right slope", right_slope
	end_time = time.time()
	timee = end_time-start_time
	print timee


	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
vs.stop()

	
