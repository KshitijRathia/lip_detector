# -*- coding: utf-8 -*-
"""
Created on Thu May 23 06:48:10 2019

@author: kshitij
"""
import cv2
import numpy as np
import dlib

facial_landmark={'lip':(48,61)}
color=(0,255,0)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

# creating models
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

video=cv2.VideoCapture(0)

while True:
    
    check,frame=video.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detecting faces
    rects=detector(gray_img,1)
    
    for i,rect in enumerate(rects):
        #search for features and store their coordinates
        shape=predictor(gray_img,rect)
        shape=shape_to_np(shape)
        
        for (name,(i,j)) in facial_landmark.items():
            # marking facial landmark and applying color
            hull=cv2.convexHull(shape[i:j])
            output=cv2.drawContours(frame,[hull],-1,color,-1)
            
    # display the result        
    cv2.imshow('capture',output)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
    
    