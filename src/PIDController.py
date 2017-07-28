from __future__ import division
import cv2
import numpy as np
import time
import imutils

class PIDController:
	def __init__(self):
		self.Kp = 0.3
		self.Kd = 0.075
		self.Ki = 0
		#self.setPoint = 0
		self.previousError = None
		self.previousTime = None
		self.integral = 0

	def setPreviousError(self,error):
		self.previousError = error

	def getPreviousError(self):
		return self.previousError

	def setPreviousTime(self,time):
		self.previousTime = time

	def compute(self, setpoint, measuredVal):
		if self.previousTime is None:
			self.previousTime = time.time()
			timeChange = 0.07
		else:
			currTime = time.time()
			timeChange = time.time() - self.previousTime

		error = setpoint-measuredVal

		if self.previousError is None:
			self.setPreviousError(0)
			
		print "error", error				
		print "previous error:",self.getPreviousError()
		
		derivative = (error - self.previousError) / timeChange
		print "derivative", derivative

		currTime = time.time()
		#timeChange = time.time() - self.previousTime
		print "timechange", timeChange
		self.integral += error * timeChange
		#print "integral", self.integral
		self.setPreviousError(error)
		self.setPreviousTime(currTime)
		
		#print "integral val:", self.Ki * self.integral
		#print "proportional val:",self.Kp*error
		#print "derivative val:", self.Kd*derivative
	
		output = self.Kp*error + self.Kd*derivative + self.Ki*self.integral

		if output > 20:
			output = 20
		elif output < -20:
			output = -20
		
		print "output", output
		return output
	
	
