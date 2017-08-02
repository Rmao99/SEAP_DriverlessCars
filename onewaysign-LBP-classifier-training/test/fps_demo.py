from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100, help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default = 1, help= "Whether or not frames should be displayed")
args = vars(ap.parse_args())

print("[INFO] sampling frames from webcam...")
stream = cv2.VideoCapture(0)
fps = FPS().start()

while fps._numFrames < args["num_frames"]:
	#grab frame from stream and resize it to have a max width of 400 pixels
	(grabbed, frame) = stream.read()
	frame = imutils.resize(frame, width=400)

	if args["display"] >0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

stream.release()
cv2.destroyAllWindows

print("[INFO] sampling frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

while fps._numFrames < args["num_frames"]:
	#grab frame from stream and resize it to have a max width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	if args["display"] >0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows
vs.stop()
