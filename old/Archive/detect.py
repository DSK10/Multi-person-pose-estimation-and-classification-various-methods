import numpy as np
import cv2,time

class Detect():
    def __init__(self,config="yolov3.cfg",weights="yolov3.weights"):
        self.CONFIDENCE = 0.2
        self.SCORE_THRESHOLD = 0.2
        self.IOU_THRESHOLD = 0

        self.config_path = config #yolov3 config file
        self.weights_path = weights #yolov3 pretrained weights
        self.labels = open("coco.names").read().strip().split("\n") #class names
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

    def detectYolo(self,image):
        Oimage = image
        image = image


        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)


        self.net.setInput(blob)
        # get all the layer names
        ln = self.net.getLayerNames()
        try:
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
        # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
            ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # feed forward (inference) and get the network output
        # measure how much it took in seconds
        start = time.perf_counter()
        layer_outputs = self.net.forward(ln)


        font_scale = 1
        thickness = 1
        boxes, confidences, class_ids = [], [], []
        # loop over each of the layer outputs
        for output in layer_outputs:
        # loop over each of the object detections
            for detection in output:
                # extract the class id (label) and confidence (as a probability) of
                # the current object detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # discard out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.SCORE_THRESHOLD, self.IOU_THRESHOLD)
        # xI = np.full(image.shape,0)
        # xI[:] = (0,100,0)
        croppedImages = []
        if len(idxs) > 0:
        # loop over the indexes we are keeping
            
            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                # draw a bounding box rectangle and label on the image
                if class_ids[i] == 0:
                    color = [int(c) for c in self.colors[class_ids[i]]]
                    # cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                    text = f"{self.labels[class_ids[i]]}: {confidences[i]:.2f}"
                    overlay = image.copy()
                    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)



                    # cv2_imshow(image)
                    # x,y,w,h = [181, 94, 104, 280] #x,y,w,h -> y:y+h,y+w:y+x
                    croppedImages.append([x,y,w,h])
                    # xI[y:y+h,x:x+w] = Oimage[y:y+h,x:x+w]
            # cv2_imshow(xI)
        return croppedImages