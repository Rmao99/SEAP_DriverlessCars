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


stop_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')
one_cascade = cv2.CascadeClassifier("oneway_classifier.xml")
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

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
		ones = one_cascade.detectMultiScale(gray,1.3,6)

		if len(stops) > 1 or len(ones) > 1:
			print "ey lmao"
		else:
			print "no stops or ones found"

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
		enc_tgt(1,1,40)
		while read_enc_status():
			frame = vs.read()
			frame = imutils.resize(frame,width=320)	
			detector.process(frame)
			x1 = detector.get_x1()
			x2 = detector.get_x2()
			print x2
			if x2 is None:
				stop()
				time.sleep(1.0)
				disable_encoders()
				enable_encoders()
				enc_tgt(1,1,18)
				while read_enc_status():
					fwd()
				disable_encoders()
				break	

			difference = 345-x2 #274 is x coord or approximate center
			power = PIDController.compute(345,x2)
			DrivePID(difference,abs(power))
		print "just broke"
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
				difference = 345-x2 #274 is x coord or approximate center
				power = PIDController.compute(345,x2)
				DrivePID(difference,abs(power))
		enable_encoders()
		set_speed(68)
		enc_tgt(1,1,8)
		while read_enc_status():
			print "in reading encorder status"
			right_rot()
		stop()		
		disable_encoders()	
		
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
