import os
from PIL import Image
import numpy as np
import cv2
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir, 'images')

front_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('jpeg') or file.endswith('JPG') or file.endswith('png'):
            path = os.path.join(root, file)
            label = os.path.basename(root).lower()
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label) # some number
            # x_train.append(path) # verify this img, turn into numpy array, convert to gray
            pil_image = Image.open(path).convert("L") # grayscale
            size = (800,800)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            img_array = np.array(final_image, 'uint8')
            # print(final_image.size)
            # print(img_array)
            front_faces = front_face_cascade.detectMultiScale(img_array, scaleFactor=1.7, minNeighbors=4)

            for (x,y,w,h) in front_faces:
                roi = img_array[y:y+h, x:x+w]
                # print(roi.size)
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)
# print(label_ids)
# print(np.array(y_labels))
# wb -> writing bytes

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml')