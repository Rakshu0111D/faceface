import os
import cv2
import numpy as np
from PIL import Image
imagepath = 0
recognizer = cv2.createLBPHFaceRecognizer();
path = 'dataSet'
def getimageswithid(path):
    imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagepath in imagepaths:
        faceimg = Image.open(imagepath).convert('L');
        facenp = np.array(faceimg,'uint8')
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(facenp)
        IDs.append(ID)
        cv2.imshow('training' , facenp)
        cv2.waitKey(10)
    return np.array(IDs),faces
IDs , faces = getimageswithid(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
