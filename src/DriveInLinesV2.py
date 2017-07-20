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
	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
	
	x1 = detector.get_x1()
	x2 = detector.get_x2()
	print "x1:",  x1
	print "x2:", x2

	if state == State.STRAIGHT or state == State.TURN_RIGHT:
		if x1 is not None and x2 is not None:
			print "found x"
			avg = (x2+x1)/2
			width = width/2
			difference = avg-width

			power = PIDController.compute(difference)
	#		print "difference in x:", difference
			DrivePID(difference,abs(power))
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
				print state
				frame = vs.read()
				frame = imutils.resize(frame,width=320)	
				detector.process(frame)
				x1 = detector.get_x1()
				x2 = detector.get_x2()
				if x2 is None:
					continue

				difference = x2-274 #274 is x coord or approximate center
				power = PIDController.compute(difference)

				DrivePID(difference,abs(power))
			stop()
			set_speed(50)
			frame = vs.read()
			frame = imutils.resize(frame,width=320)
			detector.process(frame)
			x1 = detector.get_x1()
			x2 = detector.get_x2()
			while x1 or x2 is None:
				print state
				right_rot()
				frame = vs.read()
				frame = imutils.resize(frame,width=320)
				detector.process(frame)
				x1 = detector.get_x1()
				x2 = detector.get_x2()
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
