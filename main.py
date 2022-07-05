# TechVidvan Human pose estimator
# import necessary packages

import cv2

import numpy as np
import glob
from Custommodel import ModelTrain
import torch
from poseEst import PoseEstimation
from old.Archive.detect import Detect



classModel = ModelTrain()
classModel.load_state_dict(torch.load("weight.pth"))

pse = PoseEstimation()
yolo = Detect()


# create capture object

def videoEstimation():
    inp = input("Enter 0 for webcam or path of the video : \n")
    if inp == "0":
        inp = 0

    cap = cv2.VideoCapture(inp)


    while cap.isOpened():

        _, frame = cap.read()

        
        try:
        
            frame = pse.multiDetectPose(frame,classModel)
            # frame = pse.poseDetect(frame,classModel)  --> uncomment to check for single person detection

            # or
            
            # box = yolo.detectYolo(frame)
            # frame = pse.poseDetectYOLO(frame,box,classModel) --> uncomment to check YOLO working

            cv2.imshow('Output', frame)
        except:
            cv2.imshow('Output', frame)
        
       
        
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

videoEstimation()


# for static images works best as no tracking is required

# filename = "inputs/MP.jpg"
# out = filename.split(".")[:-1]
# img = cv2.imread(filename)
# img = pse.multiDetectPose(img,classModel)

# cv2.imwrite(f"{out[0]}_output.png",img)