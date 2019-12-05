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


def getDistanceToPoint(points, FOCAL_LENGTH=399.9745178222656, CAMERA_BASELINE=0.2090607502):
    ''' Calculates the distance to the point provided in the left image '''
    pointL, pointR = points
    disparity = np.power(
        np.power(pointL[0]-pointR[0], 2) + np.power(pointL[1]-pointR[1], 2), 0.5)
    distance = FOCAL_LENGTH * CAMERA_BASELINE / disparity
    return distance


def getDistanceToBox(disparityMap, box, FOCAL_LENGTH=399.9745178222656, CAMERA_BASELINE=0.2090607502):
    ''' Calculates the distance to the box provided '''
    left = max(
        box[0], 175)  # 175 removes the zero bar on the left of disparity
    right = box[0] + box[2]
    top = box[1]
    bottom = top + box[3]

    croppedDisparity = disparityMap[top:bottom, left:right]
    # i.e. if box is empty or all zero
    if croppedDisparity.size == 0 or not np.any(croppedDisparity):
        return float("NaN")

    mode = np.apply_along_axis(lambda x: np.bincount(
        x).argmax(), axis=0, arr=croppedDisparity.flatten())

    distanceToObject = FOCAL_LENGTH * CAMERA_BASELINE / mode
    return distanceToObject


def calculateDisparity(leftImage, rightImage):
    ''' Returns array of disparities calculated with respect to the left image.'''
    images = preprocess([leftImage, rightImage])
    disparities = computeLeftAndRightDisparities(*images)
    disparity = postprocess(*disparities, originalLeftImage=leftImage)
    return disparity


def preprocess(images):
    ''' Performs a range of preprocessing actions to the image to improve results '''
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
    ''' Computte disparities with respect to both left and right images '''
    stereoProcessorR = cv2.ximgproc.createRightMatcher(stereoProcessorL)
    disparityL = stereoProcessorL.compute(leftImage, rightImage)
    disparityR = stereoProcessorR.compute(rightImage, leftImage)
    return disparityL, disparityR


def postprocess(leftDisparity, rightDisparity, originalLeftImage):
    ''' Post processing on disparity map for more accurate readings '''
    disparity = weightedLeastSquareFilter.filter(
        leftDisparity, originalLeftImage, None, rightDisparity)

    noiseFilter = 5
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - noiseFilter)

    _, disparity = cv2.threshold(
        disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity = (disparity / 16.).astype(np.uint8)

    return disparity
