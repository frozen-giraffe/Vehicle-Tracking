import json
import joblib
import random
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.utils import shuffle
import params as p
import helpers as h


# Extract the car subimages from training image using provided bboxes
def extract_ground_truth_car(folder, index):
    # imgPath = 'G:/新南研究生内容/COMP9517 CP Vision/Individual&Group_Project/Individual_Project_Data/' + folder +
    # '/clips/' + str(index) + '/imgs/040.jpg'
    imgPath = './' + folder + '/clips/' + str(index) + '/imgs/040.jpg'
    img = cv.imread(imgPath)
    # img = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    with open('./' + folder + '/clips/' + str(index) + '/annotation.json') as f:
        bboxes = json.load(f)

    cars = []

    for item in bboxes:
        bbox = item['bbox']
        top = int(bbox['top'])
        bottom = int(bbox['bottom'])
        left = int(bbox['left'])
        right = int(bbox['right'])
        car = img[top:bottom, left:right]
        cars.append(car)

    return cars


# Generate a random subimage to be used as negative labels during training
def gen_negative_sample(folder, index, size, horizon):
    imgPath = './' + folder + '/clips/' + str(index) + '/imgs/040.jpg'
    img = cv.imread(imgPath)
    # img = cv.cvtColor(img, cv.COLOR_BGR2HLS)

    rand_y = random.randint(horizon, img.shape[0] - size)  # ?
    rand_x = random.randint(0, img.shape[1] - size)

    subimage = img[rand_y:rand_y + size, rand_x:rand_x + size]
    print("subimage shaple:",subimage.shape)
    return subimage


def train(*args):
    positive_samples = []
    negative_samples = []

    print("Performing HOG feature extraction...")
    start = time.time() \
 \
    # For each image, extract the ground truth car as well as 1 random negative sample
    for i in range(1000):

        # Get ground truth car, and store the hog image of it
        cars = extract_ground_truth_car('benchmark_velocity_train', i + 1)
        for car in cars:
            hog_img = h.hog_extraction(car, p.hog_params)
            print("posi:", hog_img.shape)
            positive_samples.append(hog_img)

        # Get a negative sample and store the hog image of it
        subimage = gen_negative_sample('benchmark_velocity_train', i + 1, p.subimage_params['subimage-size'],
                                       p.subimage_params['horizon'])
        hog_img = h.hog_extraction(subimage, p.hog_params)
        print("neg:",hog_img.shape)
        negative_samples.append(hog_img)

    end = time.time()
    elapsed = divmod((end - start), 60)
    print("process time", str(int(elapsed[0])), "min,", str(int(elapsed[1])), "sec")

    # Label of 1 for positive, and -1 for negative
    positive_labels = np.ones(len(positive_samples))
    negative_labels = np.ones(len(negative_samples)) * -1

    samples = positive_samples + negative_samples
    unscaled_train = []

    # Resize each sample to be the same size then flatten to vectors
    for sample in samples:
        resized = cv.resize(sample, (80, 80))
        flattened = resized.flatten()
        unscaled_train.append(flattened)

    unscaled_train = np.asarray(unscaled_train)

    # Scale the training set to reduce model going to extremes
    scaler = StandardScaler()
    features = scaler.fit_transform(unscaled_train)
    labels = np.concatenate((positive_labels, negative_labels))

    features, labels = shuffle(features, labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                        random_state=random.randint(1, 100))

    print("X Train:", x_train.shape)
    print("Y Train:", y_train.shape)
    print("X Test:", x_test.shape)
    print("Y Test:", y_test.shape)

    svc = LinearSVC()  # test on SVC() LS minimize square hinge loss, S minimize hinge loss
    svc.fit(x_train, y_train)

    accuracy = svc.score(x_test, y_test)
    print("Accuracy:", accuracy)

    joblib.dump(svc, 'svm.pkl', compress=1)
    joblib.dump(scaler, 'scaler.pkl', compress=1)

    print("Model and Scaler saved to svm.pkl and scaler.pkl respectively")


train()
