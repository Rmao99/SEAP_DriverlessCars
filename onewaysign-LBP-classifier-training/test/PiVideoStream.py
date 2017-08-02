from picamera.array import PiRGBArray
from picamera import PiCamera 
from threading import Thread
import cv2

class PiVideoStream:
	def __init__(self,resolution=(320,240), framerate=32):
		self.camera = PiCamera()
		self.camera.resolution = resolution
		self.camera.framerate = framerate
		self.rawCapture = PiRGBArray(self.camera, size = resolution)
		self.stream = sef.camera.capture_continuous(self.rawCapture,format="bgr", use_video_port=True)

	#initialize the frame and the variable used to indicate if vid stopped
	self.frame = None
	self.stopped = False

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		#loop until stopped
		for f in self.stream:
			#grab and clear for next frame
			self.frame=f.array
			self.rawCapture.truncate(0)
			
			#if the thread indicator variable is set, stop thread
			if self.stopped:
				self.stream.close()
				self.rawCapture.close()
				self.camera.close()
				return

	def read(self):
		return self.frame

	def stop(self):
		self.stopped=True

