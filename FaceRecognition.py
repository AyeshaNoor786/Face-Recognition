import cv2
import numpy as np
import face_recognition as fr
import os

# Create a list to store images
path='ImagesBasics'
images=[]
imageNames=[]
myList=os.listdir(path)
# print(myList)
for lis in myList:
    currentImage = cv2.imread(f'{path}/{lis}')
    images.append(currentImage)
    imageNames.append(os.path.splitext(lis)[0])
print(imageNames)

# Create encodings of images
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeimg=fr.face_encodings(img)[0]
        encodeList.append(encodeimg)
    return encodeList

encodeKnownImages=findEncodings(images)
# print(len(encodeKnownImages))
print("Encoding Completed...")

# Open Web Camera
capture=cv2.VideoCapture(0)
while True:
    success,image=capture.read()

    # Resize the images
    imageSmall=cv2.resize(image,(0,0),None,0.25,0.25)
    imageSmall=cv2.cvtColor(imageSmall,cv2.COLOR_BGR2RGB)

    faceFrame=fr.face_locations(imageSmall)
    encodeFrame=fr.face_encodings(imageSmall,faceFrame)

    # Match the faces with currently available faces
    for encodeface,facelocation in zip(encodeFrame,faceFrame):
        matches=fr.compare_faces(encodeKnownImages,encodeface)
        facedistance=fr.face_distance(encodeKnownImages,encodeface)
        print(facedistance)
        matchIndex=np.argmin(facedistance)

        if matches[matchIndex]:
            name=imageNames[matchIndex].upper()
            print(name)

            # Create Rectangle
            y1,x2,y2,x1=facelocation
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(image,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(image,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        else:
            name="Not Found"
            y1,x2,y2,x1=facelocation
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(image,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(image,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Web Camera',image)
    cv2.waitKey(1)