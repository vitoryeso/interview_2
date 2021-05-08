import numpy as np
import cv2 as cv
import tensorflow as tf
import imutils

from sklearn.model_selection import train_test_split
from PIL import Image

file_names = []

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
#model = tf.keras.models.load('saved_models/' + model_name)

"""
for image in images:
    print(model.predict(image[tf.newaxis, ...]))
    vai retornar um vetor de probabilidades
    privavelmente vai ser o primeiro elem o maior prob
"""

"""
    if rotated_left:
        rotate 90 graus right

    if rotated_right
        rotate -90 graus right
    
    if cabeca pra baixo:
        rotate 180 graus
"""










