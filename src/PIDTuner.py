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

vs = PiVideoStream().start()
time.sleep(2.0)
detector = LaneDetector()

def DrivePID(difference,power):
	if difference <= 0:
		diff =int(45+power)
		print "Setting right to: ",diff
		set_left_speed(diff)
		set_right_speed(45)	
		fwd()		
	elif difference > 0:
		diff = int(45+power)
		print "Setting left to:", diff
		set_left_speed(45)
		set_right_speed(diff)
		fwd()
	else:
		print "Same speed"
		set_right_speed(45)
		set_left_speed(45)
		fwd()
PIDController = PIDController()
stop_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')

while(1):
	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	#frame = imutils.resize(frame,width=400)	

	#frame = cv2.imread("lane1_alt2")
	frame = imutils.resize(frame,width=320)	
	#frame = cv2.resize(frame,(320,240))
	detector.process(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
	tops = stop_cascade.detectMultiScale(gray, 1.3, 5)
	width = 320
	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
	
	x1 = detector.get_x1()
	x2 = detector.get_x2()
	print "x1:",  x1
	print "x2:", x2

	if x1 is not None and x2 is not None:
		print "found x"
		avg = (x2+x1)/2
		width = width/2
		difference = avg-width

		power = PIDController.compute(difference)
	#	print "difference in x:", difference
		DrivePID(difference,abs(power))
	else:
		print "missing a lane atm, stopping"
		stop()



