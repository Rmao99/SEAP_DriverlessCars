import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(1):

	_, frame = cap.read()
	
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	
	lower_blue = np.array([110,50,50])
	upper_blue = np.array([130,255,255])

	mask_wy = cv2.inRange(hsv,lower_blue,upper_blue)
	final = cv2.bitwise_and(frame,frame,mask = mask_wy)

	gauss_gray = cv2.GaussianBlur(mask_wy,(5,5,),0)
		
	cannied = cv2.Canny(gauss_gray, 50,150)
	cv2.imshow('cannied',cannied)

#ROI

#select regions
	rows, cols = frame.shape[:2] #0 to 1, rows and columns
	# make a trapezoid b/c perspective
	bot_left = [cols *0.1, rows *0.95] #small col number and large row number is bottom left
	top_left = [cols * 0.3, rows * 0.5]
	bot_right = [cols * 0.9, rows *0.95]
	top_right = [cols * 0.7, rows *0.5]

	verticies = np.array([[bot_left,top_left,top_right, bot_right]], dtype=np.int32)
#filter region
	mask = np.zeros_like(cannied)
	if len(mask.shape) ==2: #if its a binary image
		cv2.fillPoly(mask, verticies, 255)
	else:
		cv2.fillPoly(mask,verticies, (255,)*mask.shape[2])
	
	img = cv2.bitwise_and(cannied,mask)
	cv2.imshow("new frame", img)

	lines = cv2.HoughLinesP(img,  1, np.pi/180, 50,np.array([]),minLineLength=20,maxLineGap=100)
	

	if lines is not None:		
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 7)
	cv2.imshow("lines", frame)


	cv2.imshow('mask', mask_wy)	
	cv2.imshow('frame',frame)
	cv2.imshow('gauss_gray', gauss_gray)
	cv2.imshow('final',final)



	
	k = cv2.waitKey(5)
	if(k==27):
		break
cv2.destroyAllWindows()
