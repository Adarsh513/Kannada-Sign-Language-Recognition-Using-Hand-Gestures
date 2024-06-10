# -*- coding: utf-8 -*-
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["ಊಟ", "ಸಮಯ", "ತರಗತಿ", "ಮನೆ", "ನಾನು", "ನೀವು", "ಇಲ್ಲ", "ಹೌದು", "ವಿದಾಯ", "ನಮಸ್ಕಾರ","ಶೌಚಾಲಯ","ಬೇಕು","ಕುಳಿತುಕೊಳ್ಳಿ","ಕಂಡುಹಿಡಿಯಿರಿ","ಆಡುತ್ತಾರೆ","ನಾ ನಿನ್ನನ್ನ ಪ್ರೀತಿಸುತ್ತೇನೆ","ಅದ್ಭುತ","ಸಮಾನ","ಸಹೋದರ","ಸರಿಯಾದ","ಗಮನಿಸಿ","ಚಿಕ್ಕದಾಗಿದೆ","ಸೋಮವಾರ","ಮಂಗಳವಾರ","ಬುಧವಾರ","ಶುಕ್ರವಾರ","ಶನಿವಾರ","ನಿನ್ನೆ","ನಾಳೆ","ನಾನು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ","ನನಗೆ ಅರ್ಥವಾಗುತ್ತಿಲ್ಲ","ಮುಗುಳ್ನಗೆ","ಹಸಿದಿದೆ","ಎಡಬದಿ","ಬಲಭಾಗದ","ಎಲ್ಲಿ","ಏನು","ಒಂದು","ಎರಡು","ಮೂರು","ನಾಲ್ಕು","ಐದು","ಆರು","ಏಳು","ಎಂಟು","ಒಂಬತ್ತು","ಹತ್ತು"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        #cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      #(x - offset+200, y - offset-50+50), (255, 0, 255), cv2.FILLED)

        #cv2.putText(imgOutput, "", (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        print(labels[index])
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 6)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)