from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
from LaneDetector import *
from PIDController import *
from gopigo import *

import time
import cv2
import numpy as np
import imutils
import sys
import random #rng

vs = PiVideoStream().start()
time.sleep(2.0)
detector = LaneDetector()

stopWidth = 3.0 #inches
stopHeight = 3.0
oneWidth = 4.0
oneHieght = 4.0
focalLength = 305.5 #need to measure

def DrivePID(difference,power): #if difference is positive, recenter to the left
	if difference >= 0:
		diff =int(45+power)
		print "Setting right to: ",diff
		set_left_speed(diff) #motors are switched
		set_right_speed(45)	
		fwd()		
	elif difference < 0:
		diff = int(45+power)
		print "Setting left to:", diff
		set_left_speed(45)
		set_right_speed(diff)
		fwd()
	
def DrivePIDSlow(difference,power):
	if difference >= 0:
		diff =int(35+power*0.77)
		print "Setting right to: ",diff
		set_left_speed(diff)
		set_right_speed(30)	
		fwd()		
	elif difference < 0:
		diff = int(35+power*0.77)
		print "Setting left to:", diff
		set_left_speed(30)
		set_right_speed(diff)
		fwd()

def calcStopDistance(width):
	return stopWidth * focalLength/ width
def calcOneDistance(width):
	return oneWidth * focalLength/width

def adjust_gamma(frame,gamma=1.0):
	#build a lookup table mapping the pixel values [0,255] to
	#their adjusted gamma values
	invGamma = 1.0/gamma
	table = np.array([((i/255.0)**invGamma)*255
		for i in np.arange(0,256)]).astype("uint8")
	return cv2.LUT(frame,table)

def driveForwardSlow(frame):
	frame = imutils.resize(frame,width=320)	
	detector.process(frame)
	width = 320
	left_slope = detector.get_left_slope()
	right_slope = detector.get_right_slope()
	
	x1 = detector.get_x1()
	x2 = detector.get_x2()
	print "x1:",  x1
	print "x2:", x2

	if x1 is not None and x2 is not None:
		avg = (x2+x1)/2
		width = width/2
		difference = width-avg
		power = PIDController.compute(width,avg)
		DrivePIDSlow(difference,abs(power))
	elif x1 is None and x2 is not None:
		difference = 345-x2
		power = PIDController.compute(345,x2)
		DrivePIDSlow(difference,abs(power))
	elif x1 is not None and x2 is None:
		difference = -25-x1
		power = PIDController.compute(-25,x1)
		DrivePIDSlow(difference,abs(power))
	else:
		fwd()

def driveEncoderCountSlow(count):
	enable_encoders()
	enc_tgt(1,1,count)
	while read_enc_status():
		frame = vs.read()
		frame = imutils.resize(frame,width=320)	
		detector.process(frame)
		width = 320
		left_slope = detector.get_left_slope()
		right_slope = detector.get_right_slope()
	
		x1 = detector.get_x1()
		x2 = detector.get_x2()
		print "x1:",  x1
		print "x2:", x2

		if x1 is not None and x2 is not None:
			avg = (x2+x1)/2
			width = width/2
			difference = width-avg
			power = PIDController.compute(width,avg)
			DrivePIDSlow(difference,abs(power))
		elif x1 is None and x2 is not None:
			difference = 345-x2
			power = PIDController.compute(345,x2)
			DrivePIDSlow(difference,abs(power))
		elif x1 is not None and x2 is None:
			difference = -25-x1
			power = PIDController.compute(-25,x1)
			DrivePIDSlow(difference,abs(power))
		else:
			set_speed(35)
			fwd()
	disable_encoders()

def driveEncoderCount(count):
	enable_encoders()
	enc_tgt(1,1,count)
	while read_enc_status():
		frame = vs.read()
		frame = imutils.resize(frame,width=320)	
		detector.process(frame)
		width = 320
		left_slope = detector.get_left_slope()
		right_slope = detector.get_right_slope()
	
		x1 = detector.get_x1()
		x2 = detector.get_x2()
		print "x1:",  x1
		print "x2:", x2

		if x1 is not None and x2 is not None:
			avg = (x2+x1)/2
			width = width/2
			difference = width-avg
			power = PIDController.compute(width,avg)
			DrivePID(difference,abs(power))
		elif x1 is None and x2 is not None:
			difference = 345-x2
			power = PIDController.compute(345,x2)
			DrivePID(difference,abs(power))
		elif x1 is not None and x2 is None:
			difference = -25-x1
			power = PIDController.compute(-25,x1)
			DrivePID(difference,abs(power))
		else:
			set_speed(45)
			fwd()
	disable_encoders()
	

stop_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')
one_cascade = cv2.CascadeClassifier("onewaysignV4.xml")
PIDController = PIDController()
stopDist = None
oneDist = None
avg_x = None
while(1):
	
	start_time = time.time()
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
		gray2 = (gray*0.3).astype(np.uint8)	
		gray2 = adjust_gamma(gray2,0.4)
		stops = stop_cascade.detectMultiScale(gray2, 1.3, 5)
		ones = one_cascade.detectMultiScale(gray,1.2,5) #The two parameters are very important. The greater the scale factor (1.1 and 1.3), the smaller the image is when searching for your target. In this case, I did not scale down the one way sign detection as much because the sign would be too small to detect. Low min neighbors results in too many false positives, but too high results in not being able to detect your target
		print "num of stops signs found",len(stops)
		dist = None
		if len(stops) > 0: #or len(ones) > 1:
			print "Found Stop"
			stop()
			time.sleep(3.0)			
			
			for (x,y,w,h) in stops:
				print "found somethin"
				width = w
				stopDist = calcStopDistance(width)
				print "DISTANCE",dist
			print stopDist
			while stopDist > 16:
				frame = vs.read()	
				driveForwardSlow(frame)	
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				gray = (gray*0.3).astype(np.uint8)	
				gray = adjust_gamma(gray,0.4)
				stops = stop_cascade.detectMultiScale(gray, 1.3, 5)
				for (x,y,w,h) in stops:
					print "found somethin"
					#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
					width = w
					stopDist = calcStopDistance(width)
					print "DISTANCE",dist
				print dist
			print "______________________EXITING WHILE ______________________"
			stopDist = None			
			stop()
			time.sleep(2.0)			
			driveEncoderCount(38)
			stop()
			time.sleep(2.0)
			
			i = random.randint(0,1)
			if i == 0:
				driveEncoderCount(18)
				enable_encoders()
				set_speed(70)
				enc_tgt(1,1,10)
				while read_enc_status():
					print "in reading encorder status"
					left_rot()
			elif i == 1:
				driveEncoderCount(36)
			stop()		
			disable_encoders()	
			continue			 
		elif len(ones) > 0: #If found a one way sign
			print "Found One-------------------------------------------------------------------------"			
			stop()
			ratio = None
			rightcnt = 0
			leftcnt = 0
			cnt = 0
			while cnt <7:
				frame = vs.read()	
				print "just read frame"
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				ones = one_cascade.detectMultiScale(gray,1.2,5)	
				for (x,y,w,h) in ones:
					cnt+=1
					cropped_frame = frame[y:y+h,x:x+w]
				
					cropped_frame = imutils.resize(cropped_frame,width=103)
					gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
					gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]
					gray= cv2.GaussianBlur(gray, (3,3), 0)
				
					cropped_frame2 = gray[0:gray.shape[0],0:7*gray.shape[1]/20]	#check one half of the image. This is less based on lighting because we aren't looking for the raw contours, rather just the number of white pixels. (lighting + thresholding and then contouring sucks)
					black = cv2.countNonZero(cropped_frame2)
					total = cropped_frame2.shape[1] * cropped_frame2.shape[0]
					white = total-black
					ratio = white/black
					if ratio is None:
						print "no ratio val foudn"

					if ratio <= 0.42: #sign is pointing left if the ratio is less than this value
						leftcnt += 1
					elif ratio > 0.42: #Sign is pointing right if the ratio is greater than this value
						rightcnt +=1
			
			if leftcnt > rightcnt: 
				driveEncoderCountSlow(43)
				stop()
				time.sleep(2.0)
				enable_encoders()
				set_speed(70)
				enc_tgt(1,1,10)
				while read_enc_status():
					right_rot() #turn left
			elif rightcnt > leftcnt: 
				driveEncoderCountSlow(48)
				stop()
				time.sleep(2.0)
				enable_encoders()
				set_speed(70)
				enc_tgt(1,1,10)
				while read_enc_status():
					print "in reading encorder status"
					left_rot()			
			else:
				print "no val"				
			stop()		
			disable_encoders()	
			continue			 
		else:
			print "no stops or ones found"
			avg = (x2+x1)/2
			width = width/2
			difference = width-avg
			power = PIDController.compute(width,avg)
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
		enc_tgt(1,1,34)
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
				enc_tgt(1,1,16)
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
			enc_tgt(1,1,16)
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
		set_speed(70)
		enc_tgt(1,1,10)
		while read_enc_status():
			print "in reading encorder status"
			right_rot()
		stop()		
		disable_encoders()	
		
	elif x2 is None and x1 is not None:
		print "No right lane, time to turn"	
		stop()		
		time.sleep(2.0)

		enable_encoders()
		enc_tgt(1,1,34)
		while read_enc_status():
			frame = vs.read()
			frame = imutils.resize(frame,width=320)	
			detector.process(frame)
			x1 = detector.get_x1()
			x2 = detector.get_x2()
			print x2
			if x1 is None:
				stop()
				time.sleep(1.0)
				disable_encoders()
				enable_encoders()
				enc_tgt(1,1,16)
				while read_enc_status():
					fwd()
				disable_encoders()
				break	

			difference = -25-x1 #274 is x coord or approximate center
			power = PIDController.compute(-25,x1)
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
#----------------------RNG STRAIGHT OR TURN-------------------------------
			'''enable_encoders()
			enc_tgt(1,1,18)
			while read_enc_status():
				frame = vs.read()
				frame = imutils.resize(frame,width=320)	
				detector.process(frame)
				x1 = detector.get_x1()
				x2 = detector.get_x2()
				print x1
				if x1 is None:
					continue
				difference = -35-x1 #274 is x coord or approximate center
				power = PIDController.compute(-35,x1)
				DrivePID(difference,abs(power))'''
		enable_encoders()
		set_speed(70)
		enc_tgt(1,1,10)
		while read_enc_status():
			print "in reading encorder status"
			left_rot()
		stop()		
		disable_encoders()	
	else:
		stop()
		print "missing a lane atm, stopping"
			
	detector.reset()
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
vs.stop()
