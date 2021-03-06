import cv2
import sqlite3
cam = cv2.VideoCapture(1)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def insertorupdate(Id, name):
    conn= sqlite3.connect("facedb.db")
    cmd = "SELECT * FROM People WHERE Id="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd = "UPDATE People SET Name ="+str(name)+"WHERE Id ="+str(Id)
    else:
        cmd = "INSERT INTO People(Id,Name) Values("+str(Id)+","+str(name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
Id=raw_input('enter your id: ')
name = raw_input("Enter Your Name")
insertorupdate(Id,name)
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        
        sampleNum=sampleNum+1
        
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #

        cv2.imshow('frame',img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    elif sampleNum>100:
        cam.release()
        cv2.destroyAllWindows()
        break
