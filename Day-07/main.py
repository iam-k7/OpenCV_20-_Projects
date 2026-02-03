import numpy as np
import imutils        #resize image
import cv2           #image processing  
import time
import os


prototxt ="D:\OpenCV_20+_Projects\Day-07\MobileNetSSD_deploy.txt"
model = "D:\OpenCV_20+_Projects\Day-07\MobileNetSSD_deploy.caffemodel"
confThres = 0.25

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")

net = cv2.dnn.readNetFromCaffe(prototxt, model)    #load model
print("Model loaded")
print("Starting video stream...")

vs = cv2.VideoCapture(0)   #start video stream
time.sleep(2.0)

vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = vs.read()   #read frame
    frame = imutils.resize(frame, width=400)   #resize frame
    (h, w) = frame.shape[:2]   #get height and width of frame

    imResizeBlob = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(imResizeBlob, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    #print(detections)
    detshape = detections.shape[2]

    for i in np.arange(0, detshape):
        confidence = detections[0, 0, i, 2]
        if confidence > confThres:
            idx = int(detections[0, 0, i, 1])
            print("ClassID:", detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)


            if startY - 15 > 15:
                y = startY - 15
            else:
                startY + 15


            cv2.putText(frame, label, (startX, startY - 15 if startY - 15 > 15 else startY + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) 
    if key == 27:
        break

vs.release()
cv2.destroyAllWindows()


