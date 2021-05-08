import numpy as np
import cv2 as cv

from sklearn.model_selection import train_test_split
from PIL import Image

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

def load():
    file_names = []
    labels = []

    with open('train.truth.csv', 'r') as f:

        while 1:
            line = f.readline()
            if line == "":
                break
            file_name, label = line.split(',')
            file_names.append(file_name)
            labels.append(label)

    file_names = file_names[1:]
    labels = labels[1:]
    labels = list(map(label_to_id, labels))

    images = []
    for file_name in file_names:
        img = cv.imread('train/' + file_name)
        images.append(img)

    images = np.array(images)

    xtrain, xtest, ytrain, ytest = train_test_split(images, labels, test_size=0.33, random_state=42)

    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)
    return (xtrain, xtest, ytrain, ytest)
