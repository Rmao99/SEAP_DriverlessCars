from threading import Thread
import cv2

class WebcamVideoStream:
	def __init__(self,src=0):
		#init cam and read first frame
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		self.stopped = False

	def start(self):
		#start thread to read frames
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		#loop until self.stopped is true
		while True:
			if self.stopped:
				return
			
			#otherwise, read the next frame from the stream
			(self.grabbed,self.frame) = self.stream.read()

	def read(self):
		# returns most recently read frame
		return self.frame

	def stop(self):
		#thread should be stopped
			self.stopped = True
