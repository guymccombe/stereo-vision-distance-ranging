import cv2
import argparse
import sys
import math
import numpy as np
from os import path, listdir

from yolo import detectObjects
from stereo_disparity import calculateDisparity, getDistanceToBox

# Path to dataset
# ** need to edit this **
pathToDataset = "dataset"
pathToLeftImages = "left-images"     # edit this if needed
pathToRightImages = "right-images"   # edit this if needed

# resolve full directory location of data set for left / right images
pathToLeftImages = path.join(
    pathToDataset, pathToLeftImages)
pathToRightImages = path.join(
    pathToDataset, pathToRightImages)

# get a list of the left image files and sort them (by timestamp in filename)
leftFileList = sorted(listdir(pathToLeftImages))


def denseStereoRanging(skipForward=""):
    pausePlayback = False
    for leftFileName in leftFileList:
        timeStart = cv2.getTickCount()

        if skipForward and not skipForward in leftFileName:
            continue
        elif skipForward and skipForward in leftFileName:
            skipForward = ""

        images = readImages(leftFileName)
        if images == None:
            print("Images failed to load. Skipping...\n")
            continue

        imgL, imgR = images

        print("Images loaded successfully.\n")
        originalInput = np.vstack((imgL, imgR))
        cv2.imshow("Input Images", originalInput)

        objects = detectObjects(imgL)
        disparityMap = calculateDisparity(imgL, imgR)

        cv2.imshow("Disparity Map", (disparityMap *
                                     (256. / 128)).astype(np.uint8)[:, 135:])

        for i in range(len(objects)):
            distance = getDistanceToBox(disparityMap, objects[i]["box"])
            objects[i]["distance"] = distance
        displayObjects(imgL, objects)
        timeTaken = ((cv2.getTickCount() - timeStart) /
                     cv2.getTickFrequency()) * 1000

        imgL = imgL[:, 135:]

        label = f"Processing time: {timeTaken:.0f}ms"
        cv2.putText(imgL, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow("Detected objects", imgL)

        key = cv2.waitKey(
            max(2, (1 - int(math.ceil(timeTaken))) * (not pausePlayback))) & 0xFF
        if (key == ord('x')):       # exit
            print("Closing..")
            break  # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparityMap)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord(' ')):     # pause (on next frame)
            pausePlayback = not(pausePlayback)

    cv2.destroyAllWindows()


def readImages(leftFilename):
    rightFilename = leftFilename.replace("_L", "_R")
    fullPathLeft = path.join(
        pathToLeftImages, leftFilename)
    fullPathRight = path.join(
        pathToRightImages, rightFilename)

    print(fullPathLeft)
    print(fullPathRight)

    if (".png" in leftFilename) and (path.isfile(fullPathRight)):
        imgL = cv2.imread(fullPathLeft, cv2.IMREAD_COLOR)
        imgL = imgL[:-175, :]  # Crop car bonnet

        imgR = cv2.imread(fullPathRight, cv2.IMREAD_COLOR)
        imgR = imgR[:-175, :]  # Crop car bonnet

        return imgL, imgR


def displayObjects(image, objects, colour=(255, 178, 50)):
    for obj in objects:

        distance = obj["distance"]

        # Check distance is reasonable
        if math.isinf(distance) or math.isnan(distance):
            return

        left = obj["box"][0]
        right = left + obj["box"][2]
        top = obj["box"][1]
        bottom = top + obj["box"][3]
        name = obj["class"]

        # Draw a bounding box.
        cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

        # construct label
        label = f"{name}: {distance:.2f}m"

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
                      (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


if __name__ == "__main__":
    denseStereoRanging(skipForward="")
    # set this to a file timestamp to start from (empty is first example - outside lab)
    # e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
