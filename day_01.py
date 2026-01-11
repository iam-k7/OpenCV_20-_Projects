# Motion Detector Project - DAY-01


import cv2
import imutils

cam = cv2.VideoCapture(0)  # change to 1 if external camera

firstFrame = None
area = 500

while True:
    ret, img = cam.read()
    if not ret:
        break

    text = "Normal"

    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    contours, _ = cv2.findContours(
        threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv2.contourArea(c) < area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion Detected"

    cv2.putText(
        img,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Camera", img)
    cv2.imshow("Threshold", threshImg)
    cv2.imshow("Frame Difference", imgDiff)

    if cv2.waitKey(1) & 0xFF == 27:
        break  # ESC to exit


cam.release()
cv2.destroyAllWindows()
