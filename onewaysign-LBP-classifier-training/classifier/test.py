import cv2
import numpy as np
import time
import imutils

from imutils.video import WebcamVideoStream

#PI VIDEO STREAM
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera


vs = PiVideoStream().start()
#vs = WebcamVideoStream(0).start()
#cap = cv2.VideoCapture(0)

time.sleep(2.0)
oneway_cascade = cv2.CascadeClassifier('cascade.xml')
#oneway_cascade2 = cv2.CascadeClassifier('onewaysignV2.xml')

while(1):
	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	#_,frame = cap.read()
	frame = imutils.resize(frame,width=400)
    #frame = cv2.imread("stop.jpg")
   # frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
    
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ones = oneway_cascade.detectMultiScale(gray, 1.2, 5)
	#ones2 = oneway_cascade2.detectMultiScale(gray, 1.3, 5)
   	#stops3 = stop_cascade.detectMultiScale(gray, 1.5, 5)
    
    	
   	cropped_frame = None
	avg_x = None
	if len(ones) == 1:
		for (x,y,w,h) in ones:
			print "found somethin"
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
			cropped_frame = frame[y:y+h,x:x+w] #crop from y to y+h, x to x+w
			print cropped_frame.shape[1] #width
			print cropped_frame.shape[0] #height
		cropped_frame = imutils.resize(cropped_frame,width=103) #CRUCIAL: HAVE TO CROP FRAME IN ORDER TO MAKE SURE IT IS THE SAME SIZE FOR COMPARING WHITE AND BLACK PIXELS
		gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
		#ret,gray = cv2.threshold(gray,127,255,0)
		gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)[1]		
		gray= cv2.GaussianBlur(gray, (3,3), 0)
		cv2.imshow("gray",gray)
		
		cropped_frame2 = gray[0:gray.shape[0],0:7*gray.shape[1]/20]	#check one half of the image. This is less based on lighting because we aren't looking for the raw contours, rather just the number of white pixels. (lighting + thresholding and then contouring sucks)
		cv2.imshow("cropped2",cropped_frame2)
		white = cv2.countNonZero(cropped_frame2)
		total = cropped_frame2.shape[1] * cropped_frame2.shape[0]
		print "number of non zero", white
		print "total num",total
		print "ratio", white/total*1.0
		#white = total-i
		#print "white",white

		#if 0.42< white/i :c
		#	print "SIGN IS POINTING RIGHT"
		#else:
		#	print "SIGN IS POINTING LEFT"
	
		
		'''contours,heirarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		Xsum = 0
		count = 0
		print "contour length",len(contours)
		for cnt in contours: #contours is all the contours. cnt is the list of points making a single contour [ [[x1,y1]],[[x2,y2]] ]
			if 200<cv2.contourArea(cnt):
				for [[x,y]] in cnt: #unpack cnt
					Xsum += x 
					count += 1.0
					cv2.drawContours(cropped_frame,[cnt],0,(0,255,0),2)
		
		avg_x = Xsum/count'''
	elif len(ones) == 0:
		print "found nothing"
	else:
		print "whoops you found 2 lol"
	
	print "The average value of x is:",avg_x

	
	
	#for (x,y,w,h) in ones2:
	#	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	end_time = time.time()
	timee = end_time-start_time
	print timee
	cv2.imshow('img',frame)
	if cropped_frame is not None:
		cv2.imshow('cropped',cropped_frame)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cv2.destroyAllWindows()
vs.stop()
