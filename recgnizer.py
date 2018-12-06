import cv2
import numpy as np
import sqlite3
cam = cv2.VideoCapture(1)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainningData.yml")
id = 0
def getprofile(id):
    conn=sqlite3.connect("facedb.db")
    cmd = "SELECT * FROM People WHERE Id="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_DUPLEX, 1,1,0,1)
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)    
        id , conf = rec.predict(gray[y:y+h,x:x+w])
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h),font,(0,0,255))
        profile = getprofile(id)
        if(profile!=None):
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]), (x,y+h+10),font,200)            
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[2]), (x,y+h+40),font,200)
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[3]), (x,y+h+80),font,200)
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[4]), (x,y+h+120),font,200)
    cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;
        cam.release()
        cv2.destroyAllWindows()
