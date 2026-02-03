# Face emotion detection using Mobile IP Address ( IP Webcam )


import urllib.request
import cv2
import numpy as np
import imutils

url = 'http://100.66.102.111:8080'

while True:
    imgpath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgpath.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)

    frame = imutils.resize(frame, width=450)
    cv2.imshow("Frame", frame)
    if ord('q')==cv2.waitKey(1):
        exit(0)
