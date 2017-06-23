import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(1):

    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,50,50])
    upper_red = np.array([8,255,255])

    mask = cv2.inRange(hsv,lower_red, upper_red)

   # res = cv2.bitwise_and(frame,frame, mask=mask)


    image, contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255,255,255),3)
    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('image',img)
    k = cv2.waitKey(5)
    if(k==27):
        break

cv2.destroyAllWindows()
