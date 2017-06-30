import numpy as np
import cv2

#cap = cv2.VideoCapture(0)

while(1):

	#ret, frame = cap.read()
	frame = cv2.imread("stoplight.png")
	output = frame.copy()
	#frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
	
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_red = np.array([0,45,45])
	upper_red = np.array([20,255,255])

	thresh = cv2.inRange(hsv,lower_red, upper_red)
	blur = cv2.blur(thresh,(5,5))
	cv2.imshow('thresh', thresh)
	#contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(frame, contours, -1, (255,0,0),3)
	
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(blur, cv2.cv.CV_HOUGH_GRADIENT, 3.0, 100)
 
	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
 
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			print  r
			print (x,y)
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
		# show the output image
		cv2.imshow("output", np.hstack([frame, output]))
		cv2.waitKey(0)
	else:
		print "no circles found"

#cv2.imshow('frame',frame)
#cv2.imshow('thresh',thresh)
# cv2.imshow('image',img)
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
