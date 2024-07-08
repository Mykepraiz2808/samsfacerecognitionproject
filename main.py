import pickle

import cv2
import face_recognition
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')  
# IMPORTING THE MODE IMAGES INTO A LIST
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in imgModeList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
#print(len(modePathList))


#LOAD THE ENCODING FILE
print("Loading Encoded file.........")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encoded file Loaded.........")

while True:
    success, img = cap.read()
    #cv2.imshow("webcam", img)
    imgS = cv2.resize(img,(0, 0), None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    #imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)

   
    cv2.imshow("Student attendance management System", imgBackground)
    cv2.waitKey(1)  
