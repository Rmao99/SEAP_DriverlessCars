from __future__ import division
import cv2
import numpy as np
import time
import imutils

class PIDController:
	def __init__(self):
		self.Kp = 0.065
		self.Kd = -0.025
		self.Ki = 0.016
		self.setPoint = 0
		self.previousError = 0
		self.previousTime = None
		self.integral = 0

	def setPreviousError(self,error):
		self.previousError = error

	def getPreviousError(self):
		return self.previousError

	def setPreviousTime(self,time):
		self.previousTime = time

	def compute(self, measuredVal):
		if self.previousTime is None:
			self.previousTime = time.time()

		error = measuredVal
		print "error", error				
		print "previous error:",self.getPreviousError()
		
		derivative = error - self.previousError
		print "derivative", derivative
		self.setPreviousError(error)

		currTime = time.time()
		timeChange = time.time() - self.previousTime
		print "timechange", timeChange
		self.integral += error * timeChange
		print "integral", self.integral
		self.setPreviousTime(currTime)
		
		print "integral val:", self.Ki * self.integral
		print "proportional val:",self.Kp*error
		print "derivative val:", self.Kd*derivative
	
		output = self.Kp*error + self.Kd*derivative + self.Ki*self.integral
		print "output", output
		return abs(output)
	
	