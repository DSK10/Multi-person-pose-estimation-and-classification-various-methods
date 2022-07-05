import wget
from os.path import exists

url = "https://pjreddie.com/media/files/yolov3.weights"

wget.download(url, 'yolov3test.weights')
