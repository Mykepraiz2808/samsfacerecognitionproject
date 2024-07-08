import os

import cv2
import face_recognition
import pickle
import os

#IMPORTING THE STUDENT IMAGES
folderPath = 'Images'
pathList = os.listdir(folderPath)
#print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #print(path)
    #print(os.path.splitext(path)[0])
    studentIds.append(cv2.imread(os.path.splitext(path)[0]))
print(studentIds)


def findEncodings(imgList):
    encodelist = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist


print("Encoding Started ...........")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
#print(encodeListKnown)


file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
print("Encoding Completed..................")

