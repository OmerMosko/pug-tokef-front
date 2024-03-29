from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import jsonify
import logging

import sys
sys.path.append(r"C:\Users\omerm\AppData\Local\Programs\Python\Python38\include\Lib\site-packages")

# Import required modules
import numpy as np
import cv2 as cv
from cv2 import cuda

cuda.printCudaDeviceInfo(0)
import math
#from dateutil.parser import parse
import datetime as dt

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 960
inpHeight = 960
modelDetector = 'frozen_east_text_detection.pb'
modelRecognition = 'crnn_cs.onnx'

# Load network
detector = cv.dnn.readNet(modelDetector)
recognizer = cv.dnn.readNet(modelRecognition)
detector.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
detector.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
recognizer.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
recognizer.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
# Create a new named window
kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
# cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
outNames = []
outNames.append("feature_fusion/Conv_7/Sigmoid")
outNames.append("feature_fusion/concat_3")
input = 'file.img1'

'''
    Text detection model: https://github.com/argman/EAST
    Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
    CRNN Text recognition model taken from here: https://github.com/meijieru/crnn.pytorch
    How to convert from pb to onnx:
    Using classes from here: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
    More converted onnx text recognition models can be downloaded directly here:
    Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
    And these models taken from here:https://github.com/clovaai/deep-text-recognition-benchmark
    import torch
    from models.crnn import CRNN
    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load('crnn.pth'))
    dummy_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)
'''

'''
crnn_cs.onnx: https://drive.google.com/uc?export=dowload&id=12diBsVJrS9ZEl6BNUiRp9s0xPALBS7kt
frozen_east_text_detection: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
'''

############ Utility functions ############

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv.getPerspectiveTransform(vertices, targetVertices)
    result = cv.warpPerspective(frame, rotationMatrix, outputSize)
    return result


def decodeText(scores):
    text = ""
    with open('alphabet_94.txt', 'r') as file:
        alphabet = file.read().replace('\n', '')
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)


def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def is_date(string, fuzzy=False):
    formats = (
    '%d %m %Y', '%d %b %Y', '%m %Y',
    '%d/%m/%Y', '%d/%b/%Y', '%m/%Y',
    '%d.%m.%Y', '%d.%b.%Y', '%m.%Y',
    '%d %m %y', '%d %b %y', '%m %y',
    '%d/%m/%y', '%d/%b/%y', '%m/%y',
    '%d.%m.%y', '%d.%b.%y', '%m.%y'
    )

    for fmt in formats:
        try:
            t = dt.datetime.strptime(string, fmt)
            return True
        except ValueError as err:
            err
    return False

def main(imput_file):

    cap = cv.VideoCapture(imput_file)
    #cap = cv.VideoCapture(args.input if args.input else 0)
    count = 0
    tickmeter = cv.TickMeter()
    while cv.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        count += 1
        # if count%24 != 0:
        #     print(f'skipped {count}')
        #     continue
        if not hasFrame:
            break

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the detection model
        detector.setInput(blob)

        tickmeter.start()
        outs = detector.forward(outNames)
        tickmeter.stop()

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)   
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[int(i)])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH

            # get cropped image using perspective transform
            if modelRecognition:
                cropped = fourPointsTransform(frame, vertices)
                #cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

                # Create a 4D blob from cropped image
                blob = cv.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
                recognizer.setInput(blob)

                # Run the recognition model
                tickmeter.start()
                result = recognizer.forward()
                tickmeter.stop()

                # decode the result into text
                wordRecognized = decodeText(result)
                cv.putText(frame, wordRecognized, (int(vertices[1][0]), int(vertices[1][1])), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 0, 0))

                app.logger.info(wordRecognized)

                if(is_date(wordRecognized)):
                    dict = {'date': wordRecognized}
                    app.logger.info("Date rcognized: " + wordRecognized)
                    print("Date rcognized: " + wordRecognized, file=sys.stdout)
                    return dict



            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                cv.line(frame, p1, p2, (0, 255, 0), 1)

        # Put efficiency information
        label = 'Inference time: %.2f ms' % (tickmeter.getTimeMilli())
        app.logger.info(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        # cv.imshow(kWinName, frame)
        tickmeter.reset()
    return {'date': "Couldn't find date"}





app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['image']
        f.save("file.jpg")
        dict = main("file.jpg")
        return jsonify(dict)

@app.route('/uploadervideo', methods = ['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['image']
        f.save("file.mp4")
        dict = main("file.mp4")
        return jsonify(dict)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
