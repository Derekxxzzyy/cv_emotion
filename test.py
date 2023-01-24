#coding = utf8
import cv2
import time
import os
import numpy as np
from pkg_resources import EmptyProvider
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt

url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = 'Jza_grmxv0wRt1wBvd5645tSUgvvviqC'
sec = 'sYgt4gwzyVGOWj7_Nbbk7FxiPMwlU6tb'

class NoFace (Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)

def detect():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while cap.isOpened():
        time_elapsed = time.time() - start_time
        ret,img = cap.read()
        grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
        faces = detector.detectMultiScale(
            grey,
            scaleFactor = 1.05,
            minNeighbors = 10
        )
        cv2.imshow('test',img)
        if ((cv2.waitKey(30) & 0xff == ord('q')) or time_elapsed > 60 ) or len(faces) :
            x = faces[0][0]
            y = faces[0][1]
            w = faces[0][2]
            h = faces[0][3]
            cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,255,0),2)
            time.sleep(1) # Keep the face on screen
            cv2.destroyAllWindows()
            output = io.BytesIO()
            img_pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            img_pil.save(output,format='JPEG')
            img_pil.show()
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
    image = detect()
    cv2.VideoCapture(0).release()
    return postimg(url,key,sec,image)

def calemo():
    rb = getemotion()
    if rb['face_num'] == 0:
        raise NoFace("There isn't a face in the camera!")
    disg = rb['faces'][0]['attributes']['emotion']['disgust']
    fear = rb['faces'][0]['attributes']['emotion']['fear']
    neut = rb['faces'][0]['attributes']['emotion']['neutral']
    sadn = rb['faces'][0]['attributes']['emotion']['sadness']
    surp = rb['faces'][0]['attributes']['emotion']['surprise']
    return [disg,fear,neut,sadn,surp]


if __name__ == '__main__' :
    try:
        count = 0
        res = [0]
        while True :
            temp = calemo()
            res = res + temp
            count = count + 1
            time.sleep(20)
            temp = []

    except Exception as e:
        print (e)

    finally :
        print (res)
        x = []
        ydisg = []
        yfear = []
        yneut = []
        ysadn = []
        ysurp = []
        sydisg = []
        syfear = []
        syneut = []
        sysadn = []
        sysurp = []
        for i in range(1,len(res)) :
            if i % 5 == 1 :
                ydisg.append(res[i])
            if i % 5 == 2 :
                yfear.append(res[i])
            if i % 5 == 3 :
                yneut.append(res[i])
            if i % 5 == 4 :
                ysadn.append(res[i])
            if i % 5 == 0 :
                ysurp.append(res[i])
        for i in range(1 , count + 1) :
            x.append(i)
        flag = False
        if ydisg != []:
            flag = True
            plt.plot(x,ydisg,'s-',color = 'r',label="disgust") 
        if yfear != []:
            flag = True
            plt.plot(x,yfear,'s-',color = 'g',label="fear") 
        if yneut != []:
            flag = True
            plt.plot(x,yneut,'s-',color = 'b',label="neutral") 
        if ysadn != []:
            flag = True
            plt.plot(x,ysadn,'s-',color = 'y',label="sadness")
        if ysurp != []:
            flag = True
            plt.plot(x,ysurp,'s-',color = 'k',label="surprise")
        if flag == True :
            plt.xlabel("count")#横坐标名字
            plt.ylabel("emotion")#纵坐标名字
            plt.legend(loc = "best")#图例
            plt.show()
        else :
            raise NoFace("No emotion data got")

