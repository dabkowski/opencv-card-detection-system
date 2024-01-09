from time import sleep

import cv2
import numpy as np
import tensorflow as tf
import os
import tensorflow_hub as hub
from fastai.vision import *
from fastai.vision.all import load_learner
from fastai.vision.all import *

AVERAGE_CONTOUR_AREA = 0
PROCESSED_CUTOUTS = 0
TOTAL_AREAS = 0

categories = (
'Ace of clubs', 'Ace of diamonds', 'ace of hearts', 'ace of spades', 'eight of clubs', 'eight of diamonds',
'eight of hearts', 'eight of spades', 'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades', 'jack of clubs', 'jack of diamonds',
'jack of hearts', 'jack of spades', 'joker', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades', 'queen of clubs', 'queen of diamonds',
'queen of hearts', 'queen of spades', 'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades', 'ten of clubs', 'ten of diamonds', 'ten of hearts',
'ten of spades', 'three of clubs', 'three of diamonds', 'threeof hearts', ' three of spades', ' two of clubs',
'two of diamonds', ' two of hearts', 'two of spades')
# URL of the object detection model we want to use
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

# Load the object detection model
model = hub.load(model_url)

# Load image model
learner = load_learner('card_classifier_update.pkl')


# makes prediction on image for what card it thinks it is
def classify_image(img):
    img = cv2.resize(img, (224, 224))  # Resize the image
    # cv2.imshow('image', img)
    print(img.shape)
    pred, idx, probs = learner.predict(img)
    return pred

def sift_feature_matching(img1, img2):
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow("as", img3)
    return matches


def preprocess_image(image):
    imgC = image.copy()

    # Converting to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Otsu's thresholding
    Retval, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Finding contours with RETR_EXTERNAL flag to get only the outer contours
    # (Stuff inside the cards will not be detected now.)
    cont, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Creating a new binary image of the same size and drawing contours found with thickness -1.
    # This will colour the contours with white thus getting the outer portion of the cards.
    newthresh = np.zeros(thresh.shape, dtype=np.uint8)
    newthresh = cv2.drawContours(newthresh, cont, -1, 255, -1)

    # Performing erosion->dilation to remove noise(specifically white portions detected of the poker coins).
    kernel = np.ones((3, 3), dtype=np.uint8)
    newthresh = cv2.erode(newthresh, kernel, iterations=6)
    newthresh = cv2.dilate(newthresh, kernel, iterations=6)

    # Again finding the final contours and drawing them on the image.
    cont, hier = cv2.findContours(newthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(imgC, cont, -1, (255, 0, 0), 2)
    #example_card = cv2.imread('5-pik-wino.png', cv2.IMREAD_GRAYSCALE)
    # Cut out and display rectangles with a specified aspect ratio
    for i, cnt in enumerate(cont):
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)

        global PROCESSED_CUTOUTS
        PROCESSED_CUTOUTS += 1

        global TOTAL_AREAS
        TOTAL_AREAS += w * h

        global AVERAGE_CONTOUR_AREA
        AVERAGE_CONTOUR_AREA = TOTAL_AREAS / PROCESSED_CUTOUTS

        # print(f"CONTOUR AREA: {cv2.contourArea(cnt)}")
        if cv2.contourArea(cnt) < AVERAGE_CONTOUR_AREA * 0.7 and cv2.contourArea(cnt) < AVERAGE_CONTOUR_AREA * 1.2:
            continue

        # Calculate the aspect ratio
        aspect_ratio = w / h

        # Specify the aspect ratio condition (adjust as needed)
        aspect_ratio_threshold = 0.8
        #print(aspect_ratio)
        cv2.rectangle(imgC, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display each rectangle separately
        card_cutout = imgC[y:y + h, x:x + w]
        cv2.imshow(f"Card {i + 1}", card_cutout)

        #card_cutout = cv2.cvtColor(card_cutout, cv2.COLOR_BGR2GRAY)

        # matches = sift_feature_matching(example_card, card_cutout)
        # print(len(matches))
        # if len(matches) > 60:  # Adjust the threshold as needed
        #     print("MATCHES")
        #     cv2.imshow(f"Matched Card {i + 1}", card_cutout)
        classified_card = classify_image(card_cutout)

        if aspect_ratio_threshold - 0.1 <= aspect_ratio <= aspect_ratio_threshold + 0.1:
            print(classified_card)
            cv2.putText(imgC, classified_card, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Showing image with contours and rectangles
    cv2.imshow("Contours and Rectangles", imgC)

def main(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        sleep(0.01)
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocess_image(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace 'your_video.mp4' with the path to your video file
    video_path = 'karty.mp4'

    main(video_path)
