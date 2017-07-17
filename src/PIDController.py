from __future__ import division
import cv2
import numpy as np
import time
import imutils

class PIDController:
	def __init__(self):
		self.Kp = 0.065
		self.Kd = -0.025
		self.Ki = 0
		self.setPoint = 0
		self.previousError = 0

	def setPreviousError(self,error):
		self.previousError = error

	def getPreviousError(self):
		return self.previousError

	def compute(self, measuredVal):
	
		error = measuredVal
		print "error", error		
		print "previous error:",self.getPreviousError()
		derivative = error - self.previousError
		print "derivative", derivative
		self.setPreviousError(error)	
		print "proportional:",self.Kp*error
		print "derivative:", self.Kd*derivative
		output = self.Kp*error + self.Kd*derivative
		print "output", output
		return output
	
	
