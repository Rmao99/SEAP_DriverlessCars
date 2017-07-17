from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from LaneDetector import *
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
while(1):
	
	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	#frame = imutils.resize(frame,width=400)	

	#frame = cv2.imread("lane1_alt2")
	frame = imutils.resize(frame,width=320)	
	#frame = cv2.resize(frame,(320,240))
	detector.process(frame)
	width = detector.get_width()
	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
	
	x1 = detector.get_x1()
	x2 = detector.get_x2()
	print "x1:",  x1
	print "x2:", x2
	
	if state == State.STRAIGHT:
		if x1 is not None and x2 is not None:
			print "found x coordinates"
			avg = (x2+x1)/2
			width = width/2
			difference = avg-width
			print difference
			if difference > 25:
				print "Setting left to 50"
				set_right_speed(48)
				set_left_speed(45)
				fwd()
			elif difference < -25:
				print "Setting right to 50" 
				set_left_speed(48)
				set_right_speed(45)
				fwd()
			else:
				print "Same spd"
				set_right_speed(45)
				set_left_speed(45)
				fwd()
		elif x1 is None and x2 is not None:
			stop()
			state = State.TURN_LEFT
			print "No left lane, time to turn"
		elif x2 is None and x1 is not None:
			stop()
			state = State.TURN_RIGHT
			print "No right lane, time to turn"
		else:
			print "Didn't find anything"
			stop()
			state=State.STOP
	elif state == State.TURN_LEFT:
			stop()			
			enable_encoders()
			enc_tgt(1,1,46)
			fwd()
			while read_enc_status():
				time.sleep(0.1)
			enc_tgt(0,1,24)
			set_left_speed(140)
			right()
			while read_enc_status():
				time.sleep(0.1)
			disable_encoders()
			state = State.STRAIGHT
		
	'''if left_slope is not None and right_slope is not None:
		print "found slopes"
		difference = right_slope + left_slope
		if difference > 0.1428:
			difference = 0.1428

		if difference < -0.1428:
			difference = -0.1428
		
		print "difference in slopes:", difference
		if difference > 0.050:
			diff = 45+int(difference*35)
			print "Setting right to: ",diff
			set_left_speed(diff)
			set_right_speed(45)	
			fwd()		
		elif difference < -0.050:
			diff = 45-int(difference*35)
			print "Setting left to:", diff
			set_left_speed(45)
			set_right_speed(diff)
			fwd()
		else:
			print "Same speed"
			set_right_speed(45)
			set_left_speed(45)
			fwd()		
	else:
		print "missing a lane atm, stopping"
		stop()'''

	detector.reset()
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
vs.stop()





