import cv2
import face_recognition
import os


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')  #
# IMPORTING THE MODE IMAGES INTO A LIST
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in folderModePath:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
#print(len(imgModeList))    

while True:
    success, img = cap.read()
    cv2.imshow("webcam", img)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    #imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0] #
   
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)  
