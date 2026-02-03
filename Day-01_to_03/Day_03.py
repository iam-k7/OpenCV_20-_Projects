import cv2
import imutils

# RED HSV ranges
redLower1 = (0, 120, 70)
redUpper1 = (10, 255, 255)

redLower2 = (170, 120, 70)
redUpper2 = (180, 255, 255)

camera = cv2.VideoCapture(0)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=400)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create masks for red
    mask1 = cv2.inRange(hsv, redLower1, redUpper1)
    mask2 = cv2.inRange(hsv, redLower2, redUpper2)
    mask = mask1 + mask2

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]

    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]),
                      int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            if radius > 50:
                print("STOP")
            else:
                if center[0] < 150:
                    print("Right")
                elif center[0] > 350:
                    print("Left")
                else:
                    print("Back")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

camera.release()
cv2.destroyAllWindows()
