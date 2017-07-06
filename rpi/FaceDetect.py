from __future__ import print_function
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100, help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

#THREADING INCREASES FPS BY 3X!! (took 3 s to process one frame... now it takes 1
print("[INFO] sampling frames from picamera module...")
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

while fps._numFrames < args["num_frames"]:

	start_time = time.time()
	#grab the frame from the stream and resize it to have a max width of 400
	frame = vs.read()
	frame = imutils.resize(frame,width=400)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	print("detecting faces")
	start_time2 = time.time()	
	faces = face_cascade.detectMultiScale(gray, 1.1,5)
	end_time2 = time.time()
	print("detected face")
	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),2)


	if args["display"] > 0:
		cv2.imshow("Frame",frame)
		key = cv2.waitKey(1) & 0xFF

	fps.update()
	end_time = time.time()
	print(end_time-start_time)
	print(end_time2-start_time2)

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
