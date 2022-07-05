import csv, cv2, torch
import mediapipe as mp
import numpy as np
from skimage.metrics import structural_similarity as ssim

BG_COLOR = (192, 192, 192) # gray
NEW_COLOR = (0, 0, 0) # gray


class PoseEstimation():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose_estimator = []
        self.pose_estimator_dim = []
        self.selected_pose_idx = 0
        self.mp_poseDict = {
            "1":mp.solutions.pose,
            "2":mp.solutions.pose,
            "3":mp.solutions.pose,
            "4":mp.solutions.pose,
            "5":mp.solutions.pose,
            "6":mp.solutions.pose,
            "7":mp.solutions.pose,
        }
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,model_complexity=2)

        # this is dont as a try to minimise flicker (not a good approach)
        self.poseDict = {
            "1":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
            "2":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
            "3":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
            "4":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
            "5":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
            "6":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
            "7":self.mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.9,enable_segmentation=True),
        }

    # to save landmark data for training and prediction

    def saveLandMarks(self,result,save=False):
        with open("data.csv", 'a') as csv_out_file:
            if save:
                csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if result is not None:
                pose_landmarks1 = [[lmk.x, lmk.y, lmk.z] for lmk in result.landmark]
                if save:
                    pose_landmarks = np.around(pose_landmarks1, 5).flatten().astype(np.str).tolist()
                    # print(pose_landmarks)
                    csv_out_writer.writerow(pose_landmarks + ["stand"])
                return np.around(pose_landmarks1, 5).flatten().astype(np.float32).tolist()
            else:
                return None


    # method to detect single person 
    def poseDetect(self,frame,model):
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        RGB.flags.writeable = False
        results = self.pose.process(RGB)
        RGB.flags.writeable = True
        Orignal = frame.copy()
        if results.pose_landmarks is not None:
            
            lnd = self.saveLandMarks(results.pose_landmarks)

        

            lnd = torch.tensor(lnd).to(torch.float32)
            pred = model(lnd).item()
   
            annotated_image = self.annoteImage(results,frame)
            _,shape = self.croppedImage(annotated_image,frame)
            x,y,w,h = shape

            cv2.putText(Orignal, self.checkPred(pred), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.checkPred(pred,color=True), 2)
            self.mp_drawing.draw_landmarks(
                Orignal, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS) 
        return Orignal

    # to crop segmented area for contour calculation later
    def annoteImage(self,results,frame,NEW_COLOR=(0,0,0)):
        annotated_image = frame
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = NEW_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        return annotated_image

    # cropping image using contours
    def croppedImage(self,annotated_image,frame):

        imgray = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 70, 255, 0)
        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)

        frame[y:y+h,x:x+w] = 0
        
        return frame,(x,y,w,h)

    # uses YOLO model to firs crop the objects and then impose pose estimation 
    def poseDetectYOLO(self,frame,box,model):
        Orignal = frame.copy()

        for c,i in enumerate(box):

            X,Y,W,H = i

            RGB = cv2.cvtColor(frame[Y:Y+H,X:X+W], cv2.COLOR_BGR2RGB)

            selectedPose = self.getPose(RGB,i,frame.shape)

            RGB.flags.writeable = False
            results = selectedPose.process(RGB)

            RGB.flags.writeable = True
            
            
            if results.pose_landmarks is not None:
                
                lnd = self.saveLandMarks(results.pose_landmarks)

            

                lnd = torch.tensor(lnd).to(torch.float32)
                pred = model(lnd).item()


                cv2.putText(Orignal, self.checkPred(pred), (X,Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.checkPred(pred,color=True), 2)
                self.mp_drawing.draw_landmarks(
                    Orignal, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS) 
        return Orignal


    # algo to avoid flickering by selecting instance according to the object similarity
    def getPose(self,RGB,box,shape):
        print(len(self.pose_estimator))

        if(len(self.pose_estimator)==0): 
            pose = self.mp_pose.Pose(min_detection_confidence=0.5,     
                        min_tracking_confidence=0.5)
            self.pose_estimator.append(pose)    
            self.pose_estimator_dim.append(box)
            return pose
        
        if(len(self.pose_estimator) > 0):
            for idx,b in enumerate(self.pose_estimator_dim):
                score = self.compareDist(b,box,shape,RGB)
                if score >= 0.9:
                    self.pose_estimator_dim[idx] = box
                    return self.pose_estimator[idx]

            pose = self.mp_pose.Pose(min_detection_confidence=0.5,     
                        min_tracking_confidence=0.5)
            self.pose_estimator.append(pose)    
            self.pose_estimator_dim.append(box)
            return pose
            
    
    # to find similarity between 2 images              
    def compareDist(self,a,b,shape,frame):
        x,y,w,h = a
        X,Y,W,H = b

        try:
            frameA = np.squeeze(frame[y:y+h,x:x+w])
            frameB = np.squeeze(cv2.resize(frame[Y:Y+H,X:X+W],(frameA.shape[1],frameA.shape[0])))
        except:
            return 0
        print(frameA.shape,frameB.shape)
        
        ssi = ssim(frameA,frameB,channel_axis=2)
        print(ssi)
        return ssi


    # orignal approach multi person detection
    def multiDetectPose(self,frame,model):
        check = True

        croppedImage = None
        orignal = frame.copy()
        c = 1

        while check:
            try:
                pose = self.poseDict[str(c)]
            except:
                pose = self.poseDict["7"]
            if croppedImage is None:
                croppedImage = frame

            RGB = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)

            if results.pose_landmarks is not None:
                lnd = torch.tensor(self.saveLandMarks(results.pose_landmarks)).to(torch.float32)
                pred = model(lnd).item()

                annotated_image = self.annoteImage(results,croppedImage)
                croppedImage,shape = self.croppedImage(annotated_image,croppedImage)
                x,y,w,h = shape

                cv2.putText(orignal, self.checkPred(pred), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.checkPred(pred,color=True), 2)
                self.mp_drawing.draw_landmarks(
                orignal, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            else:
                check = False
            c += 1
            
            

        return orignal

    # acts as Sigmoid returns 0 or 1
    def checkPred(self, pred, color=False):
        if pred < 0.7:
            if color:
                return (36,255,12)
            return "stand"
        else:
            if color:
                return (36,36,255)
            return "squat"








        