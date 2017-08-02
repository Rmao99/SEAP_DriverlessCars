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
state = State.STRAIGHT
cnt = 0
#fwd()

def DrivePID(difference,power):
	if difference >= 0:
		diff =int(45+power)
		print "Setting right to: ",diff
		set_left_speed(diff)
		set_right_speed(45)	
		fwd()		
	elif difference < 0:
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
while(1):
	
	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	#frame = imutils.resize(frame,width=400)	

	#frame = cv2.imread("lane1_alt2")
	frame = imutils.resize(frame,width=320)	
	#frame = cv2.resize(frame,(320,240))
	detector.process(frame)
	width = 320

	x1 = detector.get_x1()
	x2 = detector.get_x2()
	print "x1:",  x1
	print "x2:", x2

	if state == State.STRAIGHT:
		if x1 is not None and x2 is not None:
			print "found x"
			avg = (x2+x1)/2
			width = width/2
			difference = width-avg

			power = PIDController.compute(width,avg)
	#		print "difference in x:", difference
			DrivePID(difference,power)
		elif x1 is None and x2 is not None:
			stop()
			state = State.TURN_LEFT
			print "No left lane, time to turn"
		elif x2 is None and x1 is not None:
			stop()
			state = State.TURN_RIGHT
			print "No right lane, time to turn"
		else:
			print "missing a lane atm, stopping"
			stop()
	elif state == State.TURN_LEFT:
			stop()			
			enable_encoders()
			enc_tgt(1,1,46)
			while read_enc_status():
				frame = vs.read()
				frame = imutils.resize(frame,width=320)	
				detector.process(frame)
				x1 = detector.get_x1()
				x2 = detector.get_x2()
				if x2 is None:
					continue

				difference = 272-x2 #272 is x coord or approximate center
				power = PIDController.compute(272,x2)

				DrivePID(difference,power)
			stop()
			detector.process(frame)
			x1=detector.get_x1()
			x2=detector.get_x2()
				
			if x1 is None and x2 is None:
				print "At Corner"
			else:
				print "At Center"
				set_left_speed(140)
				right()
				while read_enc_status():
				time.sleep(0.1)
			disable_encoders()
			state = State.STRAIGHT
			stop()			
			time.sleep(1.0)
	detector.reset()
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
vs.stop()


