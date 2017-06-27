import cv2
import numpy as np

def avg_slope(lines):

	left_lines = []
	left_weights = []
	right_lines = []
	right_weights = []

	if lines is not None:
		for line in lines:
			if line is not None:
				for x1,x2,y1,y2 in line:
					if x2 == x1:
						continue
					slope = (y2-y1)/(x2-x1)
					intercept = y1-slope*x1
					length = np.sqrt((y2-y1)**2+(x2-x1)**2)

					if slope < 0:
						left_lines.append((slope,intercept)) #append tuples
						left_weights.append(length)
					else:
						right_lines.append((slope,intercept))
						right_weights.append(length)

	#dot products
	left_lane = np.dot(left_weights,left_lines)/np.sum(left_weights)
	right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights)

	return left_lane,right_lane #(slope,intercept),(slope,intercept)

def make_points(y1,y2,line):

	if line is None:
		return None

	slope, intercept = line

	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)

	return ((x1,y1),(x2,y2))

def make_lanes(image,lines):
	#uses the avg slope method  
	(left_lane, right_lane) = avg_slope(lines)

	y1=image.shape[0] #image.shape returns the height,width, and channels (for binary images, returns just height,width)
	y2 = y1*0.6
	#y1 is very bottom of the image (higher values of y mean lower position) and y2 is slightly below the center of the image

	left_line = make_points(y1,y2,left_lane) #turn the line equations into tuples in order to draw
	right_line = make_points(y1,y2,right_lane)

	return left_line,right_line

def draw_lanes(image, lines, color = [255,0,0], thickness = 20):

	# make a separate image and combine later side by side
		for line in lines:	
			if line is not None:
				(x1,x2),(y1,y2) = line
				cv2.line(image, (x1,y1),(x2,y2), color, thickness)			
	#			cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 7)


cap = cv2.VideoCapture(0)
while(1):

	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	gauss_gray = cv2.GaussianBlur(gray,(5,5,),0)
	cv2.imshow('frame',frame)

	cannied = cv2.Canny(gauss_gray, 50,150)
	cv2.imshow('cannied',cannied)

	lines = cv2.HoughLinesP(cannied,  1, np.pi/180, 35,np.array([]),minLineLength=30,maxLineGap=100)
	cv2.imshow("lines", frame)

	draw_lanes(frame,make_lanes(frame,lines))
	cv2.imshow("new lines??", frame)

	
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()

	#line_img = np.zeros(cannied.shape, dtype=np.uint8)
'''	if lines is not None:		
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 7)'''
	

	

#AVG OUT#


# if slope of left side is steeper, move  to the right,
# if slope of right side is steeper, move to the left

'''hsl = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower = np.uint8([0,200,0])
	upper = np.uint8([255,255,255])
	mask_w = cv2.inRange(hsl,lower,upper)

	lower = np.uint8([15,100,100])
	upper = np.uint8([40,255,255])
	mask_y = cv2.inRange(hsl,lower,upper)

	mask_wy = cv2.bitwise_or(mask_w,mask_y)
	
	final = cv2.bitwise_and(frame, frame, mask = mask_wy)
	gauss_gray = gaussian_blur(gray,5)
	cv2.imshow('frame',frame)
	cv2.imshow('gray',gray)
	cv2.imshow('final',final)
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()'''

