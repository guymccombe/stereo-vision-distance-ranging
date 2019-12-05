import cv2
from numpy import vstack, median
from os import path, listdir
from math import isinf, isnan, ceil

from dense_stereo_ranging import readImages
from yolo import detectObjects
from stereo_disparity import getDistanceToPoint

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

featurePointDetector = cv2.ORB_create(nfeatures=5000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def sparseRanging(skipForwardTo=""):
    pausePlayback = False
    for leftFileName in leftFileList:
        timeStart = cv2.getTickCount()

        if skipForwardTo and not skipForwardTo in leftFileName:
            continue
        elif skipForwardTo and skipForwardTo in leftFileName:
            skipForwardTo = ""

        images = readImages(leftFileName)
        if images == None:
            print("Images failed to load. Skipping...\n")
            continue

        imgL, imgR = images

        print("Images loaded successfully.\n")
        originalInput = vstack((imgL, imgR))
        cv2.imshow("Input Images", originalInput)

        featurePointsL, descriptorsL = featurePointDetector.detectAndCompute(
            imgL, mask=None)

        featurePointsR, descriptorsR = featurePointDetector.detectAndCompute(
            imgR, mask=None)

        featureImg = cv2.drawKeypoints(imgL, featurePointsL, None)
        cv2.imshow("Features", featureImg)

        matches = matcher.knnMatch(
            descriptorsR, trainDescriptors=descriptorsL, k=2)
        goodMatches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches += [m]

        featurePointDict = {}
        for match in goodMatches:
            featurePointDict[featurePointsL[match.trainIdx]
                             .pt] = featurePointsR[match.queryIdx].pt

        objects = detectObjects(imgL)
        calculateAndDrawDistances(objects, featurePointDict, imgL)

        # Crop to remove area which cannot be ranged
        # i.e. not seen by the right camera
        imgL = imgL[:, 135:]

        # Add label displaying processing time and show image
        timeTaken = ((cv2.getTickCount() - timeStart) /
                     cv2.getTickFrequency()) * 1000
        label = f"Processing time: {timeTaken:.0f}ms"
        cv2.putText(imgL, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow("Detected objects", imgL)

        # Options to exit, save or pause
        key = cv2.waitKey(
            max(2, (1 - int(ceil(timeTaken))) * (not pausePlayback))) & 0xFF
        if (key == ord('x')):       # exit
            print("Closing..")
            break  # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord(' ')):     # pause (on next frame)
            pausePlayback = not(pausePlayback)

    cv2.destroyAllWindows()


def calculateAndDrawDistances(objs, featureDict, img):
    objsOut = []
    for obj in objs:
        box = obj["box"]
        left = box[0]
        right = box[0] + box[2]
        top = box[1]
        bottom = box[1] + box[3]

        pos = (left, right, top, bottom)
        distance = calculateDistanceToObject(*pos, featureDict)

        drawBox(img, *pos, obj["class"], distance)


def calculateDistanceToObject(left, right, top, bottom, dictionary):
    distances = []
    for pointLeft in dictionary:
        if left < pointLeft[0] < right and top < pointLeft[1] < bottom:
            pointRight = dictionary[pointLeft]
            distances += [getDistanceToPoint((pointLeft, pointRight))]

    return median(distances)


def drawBox(image, left, right, top, bottom, name, distance, colour=(255, 178, 50)):
    if isinf(distance) or isnan(distance):
        return

        # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = f"{name}: {distance:.2f}m"

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])), (left + round(
        1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


if __name__ == "__main__":
    sparseRanging()
