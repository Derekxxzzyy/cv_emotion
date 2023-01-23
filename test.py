#coding = utf8
from cairo import SurfacePattern
import cv2
import time
import os
import numpy as np
import requests
from PIL import Image
import io

url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = 'Jza_grmxv0wRt1wBvd5645tSUgvvviqC'
sec = 'sYgt4gwzyVGOWj7_Nbbk7FxiPMwlU6tb'
output = io.BytesIO()

class NoFace (Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)

def detect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,img = cap.read()
        grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        faces = detector.detectMultiScale(
            grey,
            scaleFactor = 1.05,
            minNeighbors = 15
        )
        cv2.imshow('test',img)
        if (cv2.waitKey(30) & 0xff == ord('q')) & len(faces) :
            cv2.destroyAllWindows()
            x = faces[0][0] - 100
            y = faces[0][1] + 100
            w = faces[0][2]
            h = faces[0][3]
            img_pil = Image.fromarray(cv2.cvtColor(img[x : x+w , y : y+h],cv2.COLOR_BGR2RGB))
            img_pil.save(output,format='JPEG')
            return output.getvalue()

def postimg(url,API_KEY,API_SEC,imgf):
    data = {
        'api_key': API_KEY ,
        'api_secret' : API_SEC ,
        'return_attributes' : 'emotion'
        
    }
    files = {'image_file' : imgf}
    r = requests.post(url,data = data,files = files)
    return r.json()

def getemotion():
    crop = detect()
    cv2.VideoCapture(0).release()
    return postimg(url,key,sec,crop)

if __name__ == '__main__' :
    try:
        rb = getemotion()
        if rb['face_num'] == 0:
            raise NoFace("There isn't a face in the camera!")
        ang = rb['faces'][0]['attributes']['emotion']['anger']
        disg = rb['faces'][0]['attributes']['emotion']['disgust']
        fear = rb['faces'][0]['attributes']['emotion']['fear']
        neut = rb['faces'][0]['attributes']['emotion']['neutral']
        sadn = rb['faces'][0]['attributes']['emotion']['sadness']
        surp = rb['faces'][0]['attributes']['emotion']['surprise']
        emo = [ang, disg, fear, neut, sadn, surp]
        etype = ["ang", "disg", "fear", "neut", "sadn", "surp"]
        emomain = max(emo)
        emotype = etype[emo.index(emomain)]

        print (emomain,emotype)

    except Exception as e:
        print (e)


