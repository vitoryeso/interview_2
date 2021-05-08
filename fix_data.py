import numpy as np
import cv2 as cv
import tensorflow as tf
import imutils
import pandas as pd

from sklearn.model_selection import train_test_split
from PIL import Image

file_names = []

def id_to_label(vec):
    if  vec[0]:
        return 'rotated_left'
    elif vec[1]:
        return 'upright'
    elif vec[2]:
        return 'upside_down'
    elif vec[3]:
        return 'rotated_right'

def label_to_id(string):
    if  'rotated_left' in string:
        return 0
    elif 'upright' in string:
        return 1
    elif 'upside_down' in string:
        return 2
    elif 'rotated_right' in string:
        return 3
    else:
        return None

with open('test_filenames', 'r') as f:

    while 1:
        line = f.readline()
        if line == "":
            break
        file_names.append(line.split('\n')[0])


images = []
for file_name in file_names:
    img = cv.imread('test/' + file_name)
    images.append(img)

images = np.array(images)
print(images[0].shape)
print(images[0][tf.newaxis, ...].shape)

model_name = 'keras_cifar10_trained_model.h5'
model = tf.keras.models.load_model('saved_models/' + model_name)

labels = []
for image, filename in zip(images, file_names) :
    pred = model.predict(image[tf.newaxis, ...])[0]

    labels.append(id_to_label(pred))
    if pred[0]:
        cv.imwrite('fixed/' + filename.split('.')[0] + '.png', imutils.rotate(image, -90))
    elif pred[1]:
        cv.imwrite('fixed/' + filename.split('.')[0] + '.png', image)
    elif pred[2]:
        cv.imwrite('fixed/' + filename.split('.')[0] + '.png', imutils.rotate(image, 180))
    elif pred[3]:
        cv.imwrite('fixed/' + filename.split('.')[0] + '.png', imutils.rotate(image, 90))

df = pd.DataFrame(zip(file_names, labels))
df.to_csv('fixed.csv', index=False)





