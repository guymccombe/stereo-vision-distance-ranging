'''
YOLOv3 Object Detection

Author : Toby Breckon, toby.breckon@durham.ac.uk
Refactored by Guy McCombe for use in the assignment set by Toby Breckon

Copyright (c) 2019 Toby Breckon, Durham University, UK
License : LGPL - http://www.gnu.org/licenses/lgpl.html

Implements the You Only Look Once (YOLO) object detection architecture decribed in full in:
Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
https://pjreddie.com/media/files/papers/YOLOv3.pdf

This code: significant portions based in part on the tutorial and example available at:
https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/object_detection_yolo.py
under LICENSE: https://github.com/spmallick/learnopencv/blob/master/ObjectDetection-YOLO/LICENSE

To use first download the following files:
https://pjreddie.com/media/files/yolov3.weights
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true
https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true
'''

import cv2
import numpy as np


def postprocess(image, results, threshold_confidence, threshold_nms):
    ''' Removes the bounding boxes with low confidence by non-maxima suppression '''
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    classIds = []
    confidences = []
    boxes = []
    for result in results:  # Scan through bounding boxes
        for detection in result:
            # Assign object label and confidence
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            # Only keep high confidence objects
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)


def getOutputsNames(net):
    ''' Gets names of the output layer of the CNN network '''
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


confThreshold = 0.2  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

with open("coco.names", 'rt') as f:  # Load names of classes
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


def detectObjects(frame):
    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
    tensor = cv2.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # set the input to the CNN network
    net.setInput(tensor)

    # runs forward inference to get output of the final output layers
    results = net.forward(output_layer_names)

    # remove the bounding boxes with low confidence
    classIDs, confidences, boxes = postprocess(
        frame, results, confThreshold, nmsThreshold)

    boxClassPairs = []
    for i in range(len(boxes)):
        objBox = boxes[i]
        objClass = classes[classIDs[i]]
        boxClassPairs += [{"box": objBox, "class": objClass}]

    return boxClassPairs
