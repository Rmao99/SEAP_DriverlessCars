import numpy as np
import cv2

#cap = cv2.VideoCapture(0)

while(1):

	#ret, frame = cap.read()
	frame = cv2.imread("stoplight.png")
	#frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
	
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	lower_red = np.array([0,45,45])
	upper_red = np.array([20,255,255])

	thresh = cv2.inRange(hsv,lower_red, upper_red)

	_, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, contours, -1, (255,0,0),3)

	cv2.imshow('frame',frame)
	cv2.imshow('thresh',thresh)
# cv2.imshow('image',img)
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
