import numpy as np
import cv2

#cap = cv2.VideoCapture(0)

while(1):

    #ret, frame = cap.read()
    frame = cv2.imread("stop.jpg")
   # frame = cv2.resize(frame, None, fx=3.0, fy=3.0, interpolation = cv2.INTER_CUBIC)
			    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,235])

    mask = cv2.inRange(hsv,lower_red, upper_red)

   # res = cv2.bitwise_and(frame,frame, mask=mask)


    _, contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255,0,0),3)
			
    for cont in contours:
        area = cv2.contourArea(cont)
		if area > 200:
			print "passed check, contour Area is:",area
			cnt = cont
            #perimeter = cv2.arcLength(cnt,True)
            epsilon = 0.01*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 8:
                print "yey"
                cv2.drawContours(frame, cnt, -1,(255,255,255),3)            
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
   # cv2.imshow('image',img)
    k = cv2.waitKey(5)
    if(k==27):
        break

cv2.destroyAllWindows()
