import numpy as np
import cv2
import pickle

front_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
profile_face_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

labels = {}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

capture_device = cv2.VideoCapture(0)

while True:
    ret, frame = capture_device.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = front_face_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=4)
    profiles = profile_face_cascades.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        roi_gray_res = cv2.resize(roi_gray, (800, 800))
        id_, conf = recognizer.predict(roi_gray_res)
        print(conf)
        print(labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_] + ' ' + str(round(conf, 2))
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = 'me7.png'
        cv2.imwrite(img_item, roi_color)

        color = (0, 0, 255)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    for (x, y, w, h) in profiles:
        color = (0, 0, 255)
        stroke = 2
        roi_profile_gray = gray[y:y + h, x:x + w]
        roi_profile_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow('LBPH', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture_device.release()
cv2.destroyAllWindows()
