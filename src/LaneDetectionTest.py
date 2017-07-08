from __future__ import division
import cv2
import numpy as np
#from picamera import PiCamera
#from picamera.array import PiRGBArray
import time

cap = cv2.VideoCapture(0)
#camera = PiCamera()
#camera.resolution = (640,480)
#camera.framerate = 30
#rawCapture = PiRGBArray(camera, size =(320, 240))

time.sleep(0.5)

while(1):
#for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	
	'''_, frame = cap.read()'''
	frame = cv2.imread('solidYellowLeft.jpg')
	#frame = image.array
	#rawCapture.truncate(0)	
	hsl = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
	
	lower = np.uint8([0,200,0])
	upper = np.uint8([255,255,255])
	mask_w = cv2.inRange(hsl,lower,upper)

	lower = np.uint8([10,0,100])
	upper = np.uint8([40,255,255])
	mask_y = cv2.inRange(hsl,lower,upper)

	mask_wy = cv2.bitwise_or(mask_w,mask_y)

	gray = mask_wy.copy()
	#gray = cv2.cvtColor(mask_wy, cv2.COLOR_BGR2GRAY)
	
	gauss_gray = cv2.GaussianBlur(gray,(5,5,),0)
	#cv2.imshow('gauss',gauss_gray)

	cannied = cv2.Canny(gauss_gray, 55,150)
	cv2.imshow('cannied',cannied)

#ROI

#select regions
	rows, cols = frame.shape[:2] #0 to 1, rows and columns
	# make a trapezoid b/c perspective
	bot_left = [cols *0.1, rows *0.95] #small col number and large row number is bottom left
	top_left = [cols * 0.4, rows * 0.6]
	bot_right = [cols * 0.9, rows *0.95]
	top_right = [cols * 0.6, rows *0.6]

	verticies = np.array([[bot_left,top_left,top_right, bot_right]], dtype=np.int32)
#filter region
	mask = np.zeros_like(cannied)
	if len(mask.shape) ==2: #if its a binary image
		cv2.fillPoly(mask, verticies, 255)
	else:
		cv2.fillPoly(mask,verticies, (255,)*mask.shape[2])
	
	img = cv2.bitwise_and(cannied,mask)
	cv2.imshow("new frame", img)
#HOUGH
	lines = cv2.HoughLinesP(img,  1, np.pi/180, 75,minLineLength=20,maxLineGap=300)
	

	if lines is not None:		
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 7)
	cv2.imshow("lines", frame)


#AVERAGE AND EXTRAPOLATION
	left_lines = []
	left_weights = []
	right_lines = []
	right_weights = []

	cnt = 0
	if lines is not None:
		for line in lines:
			if line is not None:
				for x1,y1,x2,y2 in line:
					print x1,y1,x2,y2
					if x2 == x1:
						continue
					slope = (y2-y1)/(x2-x1)
					intercept = y1-slope*x1
					length = np.sqrt((y2-y1)**2+(x2-x1)**2)
					print "slope",slope
					print "intercept", intercept
					#print "length", length
					if slope < -0.07:
						left_lines.append((slope,intercept)) #append tuples
						left_weights.append((length))
					elif slope >= 0.07:
						right_lines.append((slope,intercept))
						right_weights.append((length))

	#dot products
		print "left lines size:", len(left_lines)
		print "right lines size:", len(right_lines)

		left_lane = np.dot(left_weights,left_lines)/np.sum(left_weights) if len(left_weights) > 0 else None
		right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights) > 0 else None

		#dot product is slope * length + intercept*0. append (length) is equivalent to appending (length,) with no 2nd element...

		#print "left lane:", left_lane
		#print "right lane:" , right_lane
		#print "img size:",frame.shape[1]
	else:
		print "no lines found"
		continue

	if right_lane is None or left_lane is None:
		print "missing a lane, restarting"
		continue

	'''if right_lane[0] < 0.1 and right_lane[0] > -0.1:
		print "restarting"		
		continue
	if left_lane[0] < 0.1 and left_lane[0] > -0.1:
		print "restarting"		
		continue'''

	y1=frame.shape[0] #image.shape returns the height,width, and channels (for binary images, returns just height,width)

	y2 = y1*0.6

	print "left lane slope:", left_lane[0]
	print "left left intercept:",left_lane[1]
	print "y1",y1
	print "y2",y2	
	Lx1 = int((y1-left_lane[1])/left_lane[0])
	Lx2 = int((y2-left_lane[1])/left_lane[0])
	Ly1 = int(y1)
	Ly2 = int(y2)	

	left_line = ((Lx1,Ly1),(Lx2,Ly2))

	y1=frame.shape[0] #image.shape returns the height,width, and channels (for binary images, returns just height,width)
	y2 = y1*0.6

	print "right lane slope:", right_lane[0]
	print "right lane intercept:", right_lane[1]
	print "y1",y1
	print "y2",y2
	
	Rx1 = int((y1-right_lane[1])/right_lane[0])
	Rx2 = int((y2-right_lane[1])/right_lane[0])
	Ry1 = int(y1)
	Ry2 = int(y2)	
	right_line = ((Rx1,Ry1),(Rx2,Ry2))

	print "left line:", left_line
	print "right line:", right_line

	'''left_line2 = np.mean(left_line, axis = 0, dtype = np.int32)
	right_line2 =np.mean(right_line,axis=0,dtype=np.int32)

	left_line3 = tuple(map(tuple, [left_line2]))
	right_line3 = tuple(map(tuple,[right_line2]))
	print"left line 2:", left_line2
	print "right line 2:", right_line2	

	print "left line 3:",left_line3
	print "rigt line 3:",right_line3'''
	line_img = np.zeros_like(frame)
	cv2.line(line_img, (Lx1,Ly1),(Lx2,Ly2), (0,255,0), 20)
	cv2.line(line_img, (Rx1,Ry1),(Rx2,Ry2), (0,0,255), 20)

	imgg= cv2.addWeighted(frame, 1.0, line_img, 0.95,0.0)
	cv2.imshow("new lines??", imgg)

	
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
'''	for line in lines:	
		if line is not None:
			(x1,x2),(y1,y2) = line
			cv2.line(image, (x1,y1),(x2,y2), color, thickness)'''

	#draw_lanes(frame,make_lanes(frame,lines))
	
