import cv2
import argparse
import sys
import math
import numpy as np
from os import path, listdir

# Path to dataset
# ** need to edit this **
master_path_to_dataset = "C:\\Users\\guy\\Documents\\github\\stereo-vision-distance-ranging\\dataset"
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed
# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""  # set to timestamp to skip forward to

pause_playback = False  # pause until key press after each image

# resolve full directory location of data set for left / right images
full_path_directory_left = path.join(
    master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = path.join(
    master_path_to_dataset, directory_to_cycle_right)

max_disparity = 128
stereoProcessorL = cv2.StereoSGBM_create(0, max_disparity, 21)

# Parse command line arguments for the YOLO config files
parser = argparse.ArgumentParser(
    description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-r", "--rescale", type=float,
                    help="rescale image by this factor", default=1.0)
parser.add_argument("-fs", "--fullscreen", action='store_true',
                    help="run in full screen mode")
parser.add_argument("-cl", "--class_file", type=str,
                    help="list of classes", default='coco.names')
parser.add_argument("-cf", "--config_file", type=str,
                    help="network config", default='yolov3.cfg')
parser.add_argument("-w", "--weights_file", type=str,
                    help="network weights", default='yolov3.weights')

args = parser.parse_args()


def on_trackbar(val):
    ''' Dummy on_trackbar callback function '''
    return


def drawPred(image, class_name, distance, left, top, right, bottom, colour):
    ''' Draws predicted bounding box on specified image '''

    # Check distance is reasonable
    if math.isinf(distance) or math.isnan(distance):
        return

    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2fm' % (class_name, distance)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
                  (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(image, results, threshold_confidence, threshold_nms):
    ''' Remove the bounding boxes with low confidence using non-maxima suppression '''
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
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
    ''' Get the names of the output layers of the CNN network '''
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Init YOLO variables
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

classesFile = args.class_file
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNetFromDarknet(args.config_file, args.weights_file)
output_layer_names = getOutputsNames(net)

# defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


# define display window name + trackbar
windowName = 'YOLOv3 object detection: ' + args.weights_file
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName, 0, 100, on_trackbar)

cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(listdir(full_path_directory_left))

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = path.join(
        full_path_directory_left, filename_left)
    full_path_filename_right = path.join(
        full_path_directory_right, filename_right)

    # for sanity print out these filenames

    print(full_path_filename_left)
    print(full_path_filename_right)
    print()

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (path.isfile(full_path_filename_right)):

        ''' READ IMAGES '''
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgL = imgL[:-150, :]  # Crop car bonnet
        cv2.imshow('left image', imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        imgR = imgR[:-150, :]  # Crop car bonnet
        cv2.imshow('right image', imgR)

        print("-- files loaded successfully")
        print()

        ''' PERFORM YOLO ON LEFT IMAGE '''
        timeStart = cv2.getTickCount()
        frame = imgL

        if (args.rescale != 1.0):
            frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(
            frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName, windowName) / 100
        classIDs, confidences, boxes = postprocess(
            frame, results, confThreshold, nmsThreshold)

        ''' CALCULATE DISPARITY '''
        # Convert to grayscale
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Preprocessing
        # Raise to the power, as this subjectively appears to improve subsequent disparity calculation
        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        # Init WLS filter
        stereoProcessorR = cv2.ximgproc.createRightMatcher(stereoProcessorL)

        weightedLeastSquareFilter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=stereoProcessorL)
        weightedLeastSquareFilter.setLambda(80000)
        weightedLeastSquareFilter.setSigmaColor(1.2)

        # Compute disparity images
        disparityL = stereoProcessorL.compute(grayL, grayR)
        disparityR = stereoProcessorR.compute(grayR, grayL)

        # Apply WLS filtering
        disparity = weightedLeastSquareFilter.filter(
            disparityL, imgL, None, disparityR)

        # Postprocessing of disparity
        dispNoiseFilter = 5  # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

        # Scale the disparity for viewing
        _, disparity = cv2.threshold(
            disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        # Display disparity
        cv2.imshow("Disparity Map", (disparity_scaled *
                                     (256. / max_disparity)).astype(np.uint8))

        ''' DRAW DETECTED OBJECT/DISTANCE PAIRS '''
        # draw resulting detections on image
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            disparityOfObject = np.median(
                disparity_scaled[top:top + height, left:left + width])

            distanceToObject = (399.9745178222656 *
                                0.2090607502) / disparityOfObject

            drawPred(frame, classes[classIDs[detected_object]], distanceToObject,
                     left, top, left + width, top + height, (255, 178, 50))

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (
            t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image
        cv2.imshow(windowName, frame)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN & args.fullscreen)

        # stop the timer and convert to ms. (to see how long processing and display takes)
        timeEnd = ((cv2.getTickCount() - timeStart) /
                   cv2.getTickFrequency()) * 1000

        # start the event loop + detect specific key strokes
        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(max(2, 40 - int(math.ceil(timeEnd)))) & 0xFF

        # if user presses "x" then exit  / press "f" for fullscreen display
        if (key == ord('x')):       # exit
            break  # exit
        elif (key == ord('f')):
            args.fullscreen = not(args.fullscreen)
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

cv2.destroyAllWindows()
