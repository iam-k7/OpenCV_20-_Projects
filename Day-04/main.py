import cv2, os
haar_file = 'haarcase_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'Trump'

__path__ = os.path.join(datasets, sub_data)  #datasets/Kesavan
if not os.path.isdir(path):
    os.mkdir(path)
(width, height) = (130, 100)


face_casecade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)

count = 1
while count < 51:
    print(count)
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_casecade.detectMultiScale(gray, 1.3, 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0),2)
        face = gray[y:y + h, x:x + w]