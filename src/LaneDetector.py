from __future__ import division
import cv2
import numpy as np
import time
import imutils

class LaneDetector:
	def __init__(self):
		self.left_slope = None
		self.right_slope = None

	def blue(self, img): #thresholds for blue
		hsl = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

		'''lower = np.uint8([0,200,0])
		upper = np.uint8([255,255,255])
		mask_w = cv2.inRange(hsl,lower,upper)

		lower = np.uint8([10,0,100])
		upper = np.uint8([40,255,255])
		mask_y = cv2.inRange(hsl,lower,upper)

		mask_wy = cv2.bitwise_or(mask_w,mask_y)

		return mask_wy'''

		lower = np.uint8([100,50,50])
		upper = np.uint8([130,255,255])
		mask = cv2.inRange(hsl,lower,upper)

		return mask

	def applyGauss(self, gray):
		return cv2.GaussianBlur(gray,(5,5),0)

	def ROI(self, cannied):
		#select regions
		rows, cols = cannied.shape[:2] #0 to 1, rows and columns
		# make a trapezoid b/c perspective
		'''bot_left = [cols *0.1, rows *0.95] #small col number and large row number is bottom left
		top_left = [cols * 0.4, rows * 0.6]
		bot_right = [cols * 0.9, rows *0.95]
		top_right = [cols * 0.6, rows *0.6]'''

		bot_left = [0,rows]
		top_left = [0,rows*0.5]
		bot_right = [cols,rows]
		top_right = [cols,rows*0.5]

		verticies = np.array([[bot_left,top_left,top_right, bot_right]], dtype=np.int32)
	#filter region
		mask = np.zeros_like(cannied)
		if len(mask.shape) ==2: #if its a binary image
			cv2.fillPoly(mask, verticies, 255)
		else:
			cv2.fillPoly(mask,verticies, (255,)*mask.shape[2])
	
		img = cv2.bitwise_and(cannied,mask)
		return img

	def findLanes(self,frame, lines):
		left_lines = []
		left_weights = []
		right_lines = []
		right_weights = []

		cnt = 0
		if lines is None:
			print "no lines"
			return None

		for line in lines:
			for x1,y1,x2,y2 in line:
				#print x1,y1,x2,y2
				if x2 == x1:
					continue
				slope = (y2-y1)/(x2-x1)
				intercept = y1-slope*x1
				length = np.sqrt((y2-y1)**2+(x2-x1)**2)
				#print "slope",slope
				#print "intercept", intercept
				#print "length", length
				if slope < -0.082:
					left_lines.append((slope,intercept)) #append tuples
					left_weights.append((length))
				elif slope >= 0.082:
					right_lines.append((slope,intercept))
					right_weights.append((length))
	#dot products
		#print "left lines size:", len(left_lines)
		#print "right lines size:", len(right_lines)

		left_lane = np.dot(left_weights,left_lines)/np.sum(left_weights) if len(left_weights) > 0 else None
		right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights) > 0 else None

		#dot product is slope * length + intercept*0. append (length) is equivalent to appending (length,) with no 2nd element...
		if right_lane is None or left_lane is None:
			print "missing a lane, restarting"
			return None

		self.set_left_slope(left_lane[0])
		self.set_right_slope(right_lane[0])

		y1=frame.shape[0] #image.shape returns the height,width, and channels (for binary images, returns just height,width)

		y2 = y1*0.5

		#print "left lane slope:", left_lane[0]
		#print "left left intercept:",left_lane[1]
		#print "y1",y1
		#print "y2",y2	
		Lx1 = int((y1-left_lane[1])/left_lane[0])
		Lx2 = int((y2-left_lane[1])/left_lane[0])
		Ly1 = int(y1)
		Ly2 = int(y2)	

		left_line = ((Lx1,Ly1),(Lx2,Ly2))

		y1=frame.shape[0] #image.shape returns the height,width, and channels (for binary images, returns just height,width)
		y2 = y1*0.5

		#print "right lane slope:", right_lane[0]
		#print "right lane intercept:", right_lane[1]
		#print "y1",y1
		#print "y2",y2
	
		Rx1 = int((y1-right_lane[1])/right_lane[0])
		Rx2 = int((y2-right_lane[1])/right_lane[0])
		Ry1 = int(y1)
		Ry2 = int(y2)	
		right_line = ((Rx1,Ry1),(Rx2,Ry2))

		#print "left line:", left_line
		#print "right line:", right_line
	
		return left_line, right_line

	def drawLines(self,frame,lanes):
		left_line,right_line = lanes
	
		line_img = np.zeros_like(frame)
		((x1,y1),(x2,y2)) = left_line
		cv2.line(line_img, (x1,y1),(x2,y2), (0,255,0), 20)
		((x1,y1),(x2,y2)) = right_line
		cv2.line(line_img, (x1,y1),(x2,y2), (0,0,255), 20)
		return line_img

	def process(self, frame):

#		cv2.imshow("original", frame)
#		cv2.imwrite("original.png",frame)
		gray = self.blue(frame)
#		cv2.imshow('threshed',gray)
#		cv2.imwrite("threshed.png", gray)
		gauss_gray = self.applyGauss(gray)
#		cv2.imwrite('blurred.png',gauss_gray)
		cannied = cv2.Canny(gauss_gray, 55,150)
#		cv2.imshow('Edges',cannied)
#		cv2.imwrite("edges.png", cannied)
		region = self.ROI(cannied)
#		cv2.imshow("Region of Interest", region)
#		cv2.imwrite("roi.png", region)
		
		lines = cv2.HoughLinesP(region,  1, np.pi/180, 25,minLineLength=23,maxLineGap=300)

		'''copy= frame.copy()
		if lines is not None:		
			for line in lines:
				for x1,y1,x2,y2 in line:
					cv2.line(copy, (x1,y1), (x2,y2), (255,0,0), 7)
		cv2.imshow("Hough Lines", copy)'''
#		cv2.imwrite("hough.png", copy)

		lanes = self.findLanes(frame,lines)

		'''if lanes is not None:	
			line_img = self.drawLines(frame,lanes)
			final = cv2.addWeighted(frame, 1.0, line_img, 0.5,0.0)
			cv2.imshow("overlay", final)
#			cv2.imwrite("overlay.png", final)'''
		return lanes

	def get_left_slope(self):
		return self.left_slope

	def get_right_slope(self):
		return self.right_slope

	def set_left_slope(self,val):
		self.left_slope = val

	def set_right_slope(self,val):
		self.right_slope = val

	def reset(self):
		self.left_slope = None
		self.right_slope = None


	

