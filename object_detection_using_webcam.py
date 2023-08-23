import cv2;
import numpy as np;
threshold=0.5;
#using webcam
cap=cv2.VideoCapture(0);
cap.set(3,640);
cap.set(4,480);
classFile='coco.names';
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n');
print(classNames);

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt';
weightsPath='frozen_inference_graph.pb';

net=cv2.dnn_DetectionModel(weightsPath,configPath);
net.setInputSize(320,320);
#for normalizing the pixel
net.setInputScale(1.0/127.5);
#this helps to center pixel around mean each for each filter channel
net.setInputMean((127.5,127.5,127.5));
#opencv import image into rgb but model requirees bgr image
net.setInputSwapRB(True);


while True:
    success,image=cap.read();
    #only object will be detected whose confidence score greater then 0.5;
    classIds, confs, bbox = net.detect(image, threshold);
    #if no object id detected
    # if len(classIds)==0:
    #     msg="no any object";
    #     cv2.putText(image, str(msg), (40,100), cv2.FONT_HERSHEY_PLAIN, 1,
    #                 (0, 255, 0), 8)

    print(classIds,bbox);
    if len(classIds)!=0:
    #we are getting error if no any object is detected then its ruuning loss so we need to set condition
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
           cv2.rectangle(image, box, color=(0, 255, 0), thickness=3);
           cv2.putText(image, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 2)
           cv2.putText(image, str(round(confidence*100,2)), (box[0] + 150, box[1] + 30), cv2.FONT_HERSHEY_PLAIN, 1,
                       (0, 255, 0), 2)





    cv2.imshow("image",image);
    cv2.waitKey(1);

