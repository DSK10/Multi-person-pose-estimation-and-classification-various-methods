To run this first install all the packages from the requirements.txt file

it is prefered to download the weights from the given link and not from the code as there could exist certificate errors
in the given system, else it would work fine

weights are not included in the zip as file size is big

https://pjreddie.com/media/files/yolov3.weights

name of the weights should be yolov3.weights

run the main.py file

if every resources are available it will ask input
    - 0 input will take input from webcam

    - filename or filepath will read the video file

output will be realtime on a new window


NOTE : There are other methods available in the class but are not efficient as the original one, also the flickering in the pose
is due to mis-match of pose instances that can be resolved using an appropriate algorithm which can be done but I need more time on that.
    - Classification is done using Neural Network
    - Trained on only 20-25 data
    - Working fine but results might not be perfect

![MP_output](https://user-images.githubusercontent.com/38138168/177247937-d2e5a065-fc1e-4d8b-b629-46f5e5edf00b.png)
