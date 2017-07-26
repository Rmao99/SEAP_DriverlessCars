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
	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
	
	x1 = detector.get_x1()
	x2 = detector.get_x2()
	print "x1:",  x1
	print "x2:", x2

	if x1 is not None and x2 is not None:
		print "found x"
		print x1
		print x2
		avg = (x2+x1)/2
		width = width/2
		difference = width-avg

		power = PIDController.compute(width,avg)
	#	print "difference in x:", difference
		DrivePID(difference,abs(power))
	elif x1 is None and x2 is not None:
		print "No left lane, time to turn"		
		stop()		
		time.sleep(2.0)
		'''skip = False
		cnt = 0
		while(cnt < 2):
			frame = vs.read()
			frame = imutils.resize(frame,width=320)
			detector.process(frame)
			x1 = detector.get_x1()
			x2 = detector.get_x2()
			if x1 is not None:
				skip = True
			cnt+=1	

		if skip == True:
			continue'''

		enable_encoders()
		enc_tgt(1,1,50)
		while read_enc_status():
			frame = vs.read()
			frame = imutils.resize(frame,width=320)	
			detector.process(frame)
			x1 = detector.get_x1()
			x2 = detector.get_x2()
			print x2
			if x2 is None:
				continue

			difference = 274-x2 #274 is x coord or approximate center
			power = PIDController.compute(274,x2)
			DrivePID(difference,abs(power))
		
		stop()
		disable_encoders()

		time.sleep(1.0)
		frame = vs.read()
		frame = imutils.resize(frame,width=320)	
		detector.process(frame)
		x1 = detector.get_x1()
		x2 = detector.get_x2()
		if x1 is not None and x2 is not None:
			enable_encoders()
			enc_tgt(1,1,18)
			while read_enc_status():
				frame = vs.read()
				frame = imutils.resize(frame,width=320)	
				detector.process(frame)
				x1 = detector.get_x1()
				x2 = detector.get_x2()
				print x2
				if x2 is None:
					continue

				difference = 274-x2 #274 is x coord or approximate center
				power = PIDController.compute(274,x2)
				DrivePID(difference,abs(power))
		
		set_speed(50)
		enc_tgt(1,1,18)
		while read_enc_status():
			print "in reading encorder status"
			right_rot()
		disable_encoders()
		stop()			
		time.sleep(1.0)
		
	elif x2 is None and x1 is not None:
		stop()
		print "No right lane, time to turn"
	else:
		stop()
		print "missing a lane atm, stopping"
			
	detector.reset()
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
vs.stop()
