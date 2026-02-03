# Face Detection

import cv2

# Load Haar Cascade
alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haarcase_cascade = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haarcase_cascade.detectMultiScale(grayImg, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(10) == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
