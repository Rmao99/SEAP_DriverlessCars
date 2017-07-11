from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from LaneDetector import LaneDetector
from gopigo import *

import time
import cv2
import numpy as np
import imutils
import sys

vs = PiVideoStream().start()
time.sleep(2.0)
detector = LaneDetector()

fwd()
while(1):
	
	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	#frame = vs.read()
	#frame = imutils.resize(frame,width=400)	

	frame = cv2.imread("lane1_alt2")
	frame = cv2.resize(frame,(320,240))
	detector.process(frame)

	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
		
	if left_slope is not None and right_slope is not None:
		print "found slopes"
		difference = right_slope + left_slope
		difference = int(difference)
		print "difference in slopes:", difference
		if difference > 0.02:
			set_right_speed(150+difference*60)
			set_left_speed(150)			
		elif difference < -0.02:
			set_right_speed(150)
			set_left_speed(150+difference*60)
		else:
			set_right_speed(150)
			set_left_speed(150)			
	else:
		print "missing a lane atm"
	detector.reset()
	#sys.stdout.flush()
	#except:
	#stop()
	#print "gonna exit xddddddddddddddddddddddd"
	#e = sys.exc_info()[0]
	#print e
	#sys.exit()
