from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import os
from matplotlib import pyplot as plt

# launching the picamera

camera = PiCamera()
rawcapture = PiRGBArray(camera)

time.sleep(0.1)

camera.capture(rawcapture,format='bgr')
image= rawcapture.array

#image becomes the source image for putting it through the segmentation algorithm

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)

edges = cv2.Canny(blur,30,60)

im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

cv2.drawContours(edges.copy(),contours,-1,(0,255,0),3)

plt.subplot(111),plt.imshow(edges,cmap='gray'),plt.title('segmented image')
plt.xticks([]),plt.yticks([])
plt.show()

#cv2.imshow('image',image)
#cv2.waitKey(0)



