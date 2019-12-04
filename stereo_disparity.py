'''
Module for calculating stereo disparity.

Author: Guy McCombe,
Based on an example provided by Toby Breckon
'''

import cv2
import numpy as np

# Initialisation
max_disparity = 128
stereoProcessorL = cv2.StereoSGBM_create(0, max_disparity, 21)

lmbda = 80000
sigma = 1.2
weightedLeastSquareFilter = cv2.ximgproc.createDisparityWLSFilter(
    matcher_left=stereoProcessorL)
weightedLeastSquareFilter.setLambda(lmbda)
weightedLeastSquareFilter.setSigmaColor(sigma)


def getDistanceToBox(disparityMap, box, FOCAL_LENGTH=399.9745178222656, CAMERA_BASELINE=0.2090607502):
    left = box[0]
    right = left + box[2]
    top = box[1]
    bottom = top + box[3]

    croppedDisparity = disparityMap[top:bottom, left:right]

    try:
        disparityOfObject = np.percentile(
            croppedDisparity, 25, axis=0, overwrite_input=True)
        disparityOfObject = np.median(disparityOfObject)
    except:  # Fall back to mean if median throws exception
        disparityOfObject = np.mean(croppedDisparity)

    distanceToObject = FOCAL_LENGTH * CAMERA_BASELINE / disparityOfObject
    return distanceToObject


def calculateDisparity(leftImage, rightImage):
    ''' Returns array of disparities calculated with respect to the left image.'''
    images = preprocess([leftImage, rightImage])
    disparities = computeLeftAndRightDisparities(*images)
    disparity = postprocess(*disparities, originalLeftImage=leftImage)
    return disparity


def preprocess(images):
    for i in range(len(images)):
        image = images[i]
        image = convertToGrayscale(image)
        image = raiseToPower(image)
        image = equalizeBrightness(image)
        images[i] = image
    return images


def convertToGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def raiseToPower(image):
    return np.power(image, 0.75).astype(np.uint8)


def equalizeBrightness(image):
    return cv2.equalizeHist(image)


def computeLeftAndRightDisparities(leftImage, rightImage):
    stereoProcessorR = cv2.ximgproc.createRightMatcher(stereoProcessorL)
    disparityL = stereoProcessorL.compute(leftImage, rightImage)
    disparityR = stereoProcessorR.compute(rightImage, leftImage)
    return disparityL, disparityR


def postprocess(leftDisparity, rightDisparity, originalLeftImage):
    disparity = weightedLeastSquareFilter.filter(
        leftDisparity, originalLeftImage, None, rightDisparity)

    noiseFilter = 5
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - noiseFilter)

    _, disparity = cv2.threshold(
        disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity = (disparity / 16.).astype(np.uint8)

    return disparity
