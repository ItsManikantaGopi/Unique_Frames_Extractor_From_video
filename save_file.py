#file saver
import image_compare as ic
import numpy as np
import cv2
import os 
from image_compare import compare_util

def save_v1(prev_image,current_image,name_of_video):
    try:
        comp_value=ic.compare(prev_image,current_image)
    except:
        comp_value=1
    if comp_value and ic.compare(np.zeros(current_image.shape, current_image.dtype)*0,current_image):
        l=os.listdir() 
        if name_of_video not in l:
            os.mkdir(name_of_video)
        temp=len(os.listdir(name_of_video))
        imagename=name_of_video+"/"+name_of_video+"_"+str(temp)+".jpg"
        M=np.ones(current_image.shape,dtype='uint8')*30
        # added_img=cv2.add(current_image,M)
        cv2.imwrite(imagename,current_image)
  

def save(current_image, name_of_video):
    if ic.compare_util(np.zeros(current_image.shape, current_image.dtype)*0, current_image):
        l = os.listdir()
        if name_of_video not in l:
            os.mkdir(name_of_video)
        temp = len(os.listdir(name_of_video))
        imagename = name_of_video+"/"+name_of_video+"_"+str(temp)+".jpg"
        M = np.ones(current_image.shape, dtype='uint8')*30
        # added_img=cv2.add(current_image,M)
        cv2.imwrite(imagename, current_image)





