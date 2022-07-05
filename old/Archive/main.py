import cv2
import numpy as np
from detect import Detect
import os
import wget

os.mkdir('frames')

# from os.path import exists

# if not exists('yolov3.weights'):
    
#     take = input("would you like to download the yolov3 weights, press 1 to confirm press 0 to exit")
#     if take == '1':
#         url = "https://pjreddie.com/media/files/yolov3.weights"
#         wget.download(url, 'yolov3test.weights')
#         print("downloading yolov3 weights please wait, this will take a while ...")



# videoInput = input('Enter file path to process or enter 0 to use webcam : \n')

# if videoInput == '0':
#     vid = cv2.VideoCapture(0)
#     fileOutName = 'webcam'
# else:
#     vid = cv2.VideoCapture(videoInput)
#     fileOutName = videoInput

vid = cv2.VideoCapture('detectVideo.mp4')
fileOutName = 'detectVideo_output2'

c = 1
getDetect = Detect(config="yolov3.cfg",weights="yolov3.weights")

img_array = []
while(True):
    ret, frame = vid.read()
  
    height, width, layers = frame.shape
    size = (width,height)


    x = getDetect.detectYolo(frame)
    cv2.imshow('footage',frame)
    cv2.imwrite(f'frames/{c}.png',x)

    img_array.append(cv2.imread(f'frames/{c}.png'))

    c += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter(f'{fileOutName}_output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()