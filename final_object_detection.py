import sys
import cv2
import numpy as np
import time
from darkflow.net.build import TFNet
import json
import math


################################################
def WriteToJsonFile(file_path,data):
        with open(file_path,"a") as fp:
                json.dump(data,fp)


file_path = 'D:/BSM_Message.json'
#data = {'frame_id':'count','x coordinate':'cx','y coordinate':'cy'}
#WriteToJsonFile(file_path,data)
################################################
#print ('Hello world!')
version = cv2.__version__.split('.')[0]

def pixeltogeo(cx,cy):
    Lx = 34.668316 +  (0.000027/800)*cx
    Ly = -82.826304 + (0.000043/1000)*cy
    print('Lx',Lx)
    print('Ly',Ly)

def get_speed(Time,Cx,Cy):
    print('Cx :',Cx)
    print('Cy :',Cy)
    d = ((x1-Cx)*(x1-Cx) + (y1-Cy)*(y1-Cy))
    print('x1 :',x1)
    print('y1 :',y1)
    dt = math.sqrt(d)
    dist = dt*(20/1680)
    
    time = (Time-t)
    if time > 0:
        speed = (dist/time)
    else:
        speed = 0
    print('Time:',time)
    print('Distance :',dist)
    print('Speed :',speed)
    
option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.25,
    'gpu': 1.0
}

tfnet = TFNet(option)

objects = ['person']

cam = "http://192.168.0.10/JpegStream.cgi?username=BDD7AA6E7E26BE386752C3E963BDDB3A7EB41DD0B5EAE2EE903D5EFE4CD9D98C&password=BDD7AA6E7E26BE386752C3E963BDDB3A7EB41DD0B5EAE2EE903D5EFE4CD9D98C&channel=1&secret=1&key=382C76D17phndi"
#cam = 0 # Use  local webcam.

capture = cv2.VideoCapture(cam)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
if not capture:
    print("!!! Failed VideoCapture: invalid parameter!")

#check opencv version
if version == '2' :
    fgbg = cv2.BackgroundSubtractorMOG2()
if version == '3': 
    fgbg = cv2.createBackgroundSubtractorMOG2()


#data1 = data.tolist()


#path = 'D:/Eshaa'
#filename = 'bsm.json'
#data = (frame,label,cx,cy)
#data['test'] = 'test2'
#data['hello'] = 'world'



count = 0
x1=0
y1=0
t=0
Cx=0
Cy=0
Lx=0
Ly=0
dist=0
speed=0


while (capture.isOpened()):
    stime = time.time()
    print('Stime :',stime)
    ret, frame = capture.read()
    #time =0
    
    if ret:
        results = tfnet.return_predict(frame)
        #print (results)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label'] + ":" + str('%0.2f' %result['confidence'])
            

            #cx = int((result['topleft']['x'] + result['bottomright']['x'] )/2.0)
            #cy = int((result['topleft']['y'] + result['bottomright']['y'] )/2.0)
            #cx_1 = cx.tolist()
            #cy_1 = cy.tolist()
            if(result['label'] in objects ):

                cx = int((result['topleft']['x'] + result['bottomright']['x'] )/2.0)
                cy = int((result['topleft']['y'] + result['bottomright']['y'] )/2.0)
            
                frame = cv2.rectangle(frame, tl, br, (0,255,0), 4)
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.circle(frame,(cx,cy), 4, (0,0,255), -1)
               


                #_, frame = capture.read()
                
                cp_frame = frame.copy()
                #cropped = cp_frame[400:800,100:1140]
                #cv2.imshow("cropped", cropped)
                #cv2.circle(cropped,(80,200),5,(255,0,0),-1)
                #cv2.circle(cropped,(840,200),10,(0,0,255),-1)
                #cv2.circle(cropped,(60,0),15,(0,255,0),-1)
                #cv2.circle(cropped,(860,0),20,(0,0,0),-1)

                #pts1 = np.float32([[80,200],[840,200],[60,0],[860,0]])
                #pts2 = np.float32([[80,0],[840,0],[60,200],[860,200]])
       
                #matrix = cv2.getPerspectiveTransform(pts1,pts2)
                #result = cv2.warpPerspective(cropped,matrix,(0,1000))
                #both_img = cv2.flip( result, 0 )
                cv2.resize(frame,(800,1000))
                #print('cropped image size : ' , cropped.shape)

                #cv2.imshow('frame',cropped)
                #cv2.imshow('Perspective Transform',new_img)
            
                #apply background substraction
                
                        
                #check opencv version
                if version == '2' : 
                     (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if version == '3' : 
                     (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
                #looping for contours
                for c in contours:
                    if cv2.contourArea(c) < 300:
                        continue

                #get bounding box from countour
                (x, y, w, h) = cv2.boundingRect(c)

                #draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                #time = time
                #print(time)
                
                get_speed(stime,cx,cy)
                pixeltogeo(cx,cy)
        
                x1 = cx
                y1 = cy
                t = stime
                data = {'id':count,'object':label,'xpixel-cordinate':cx,'ypixel-cordinate':cy,'x-cordinate':Lx,'y-coordinate':Ly,'Distance':dist,'Speed':speed}

        fgmask = fgbg.apply(frame)
        cv2.imshow('foreground and background',fgmask)
        cv2.imshow('rgb',frame)
        

        print ('Frame id : ' , count )
        #x=0
        #y=0
        #t=0
        

                
        #WriteToJsonFile(file_path,data)
        count+=1
        
        #cv2.imshow('frame', frame)
        #print('FPS {:.1f}'.format(1 / (time.time() - stime cv2.destroyAllWindows()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    else:
        capture.release()
        cv2.destroyAllWindows()
        
        break


   
