import joblib
import random
import json
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from detect import *

def main(*args):

    try:
        model = joblib.load('svm.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception:
        print('Error: Failed to load model and scaler')
        return

    # Get a random image from the test folder
    fileNumber = random.randint(1, 270)
    path = './benchmark_velocity_test/' + str(fileNumber) + '/imgs/040.jpg'
    img = cv.imread(path, 1)
    colored = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2HLS) # Convert to HLS for input into detection system

    results = detect_vehicles(img, model, scaler)

    # For the given test image, show the produced bounding boxes with and without the correct bounding boxes on top
    img_bbox = colored
    for box in results:
        dims = box['bbox']
        img_bbox = cv.rectangle(img_bbox, (dims['left'], dims['top']), (dims['right'], dims['bottom']), (255, 0, 0), 2)

    overlap = img_bbox

    # Get the ground truth labels for the image and add them to the image
    with open('./benchmark_velocity_test/' + str(fileNumber) +'/annotation.json') as f:
        bboxes = json.load(f)
    for item in bboxes:
        bbox = item['bbox']
        top = int(bbox['top'])
        bottom = int(bbox['bottom'])
        left = int(bbox['left'])
        right = int(bbox['right'])
        overlap = cv.rectangle(overlap, (left, top), (right, bottom), (0, 255, 0), 2)


    figure(figsize=(16, 12), dpi=80)
    plt.imshow(overlap)
    plt.show()

main()