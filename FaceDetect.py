import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

while(1):

	ret,frame = cap.read()
	
	gray = cv2.cvtColo(frame,cv2.Color_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1,5)

	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),2)

	cv2.imshow("Feed", frame)

	k=cv2.waitKey(5)
	if (k==27):
		break
cv2.destroyAllWindows()
