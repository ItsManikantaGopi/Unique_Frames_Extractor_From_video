from matplotlib import pyplot as plt
import copy,glob
import math,operator
from functools import reduce
from PIL import Image 
from PIL import ImageChops
import cv2
error = 90
def remove_black_background(img):
    # keep a copy of original image
    # original = cv2.imread(IMG_IN)

    # Read the image, convert it into grayscale, and make in binary image for threshold value of 1.
    # img = cv2.imread(IMG_IN,0)

    # use binary threshold, all pixel that are beyond 3 are made white
    _, thresh_original = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)

    # Now find contours in it.
    thresh = copy.copy(thresh_original)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get contours with highest height
    lst_contours = []
    for cnt in contours:
        ctr = cv2.boundingRect(cnt)
        lst_contours.append(ctr)
    x, y, w, h = sorted(lst_contours, key=lambda coef: coef[3])[-1]

    # draw contours
    ctr = copy.copy(original)
    cv2.rectangle(ctr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    ctr = ctr[y:h, x:w]
    return ctr
def template_match(parent,template):
    res = cv2.matchTemplate(parent, template, cv2.TM_CCOEFF_NORMED)
    return
import numpy as np
def compare_hist(im1, im2):
    "Calculate the root-mean-square difference between two images"
    img=cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
    #img = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
    im1 = Image.fromarray(img)
    img=cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)
    #img = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAYY)
    im2 = Image.fromarray(img)
    #cv2.imshow("gray",img)
    #cv2.waitKey()
    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    a= math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(256))
    ) / (float(im1.size[0]) * im1.size[1]))
    #print(a)z
    return a>error
orb=cv2.ORB_create(nfeatures=1000)
def compare2(im1,im2):
	kp1,des1=orb.detectAndCompute(im1,None)
	kp2,des2= orb.detectAndCompute(im2,None)
	#print(len(des1)," - ",len(des2))
	if des1 is not None and des2 is not  None:
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
		matches=bf.match(des1,des2)
		if len(matches)>800 and len(kp1)>500 and len(kp2)>500:
			return False	
	return True


def dice_coef(y_true, y_pred):
    # y_true = tf.keras.layers.Flatten()(y_true)
    # y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = np.sum(np.multiply(y_true , y_pred))
    # print(np.sum(np.multiply(y_pred, y_true))==intersection)
    smooth=0
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)



def compare(y_true, y_pred):
    y_true=cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
    y_pred=cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
    # print(y_pred.shape)
    # print(y_true.shape)
    loss=1.0 - dice_coef(y_true, y_pred)
    # print(loss)
    if loss>0.5:
        # print(loss)
        return True 
    return False

def compare(y_true,y_pred):
    y_true = y_true/255
    y_pred = y_pred/255
    return np.sum((y_true.flatten()-y_pred.flatten())**2)
def binarize(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,111, 19)

def compare_util(imageA, imageB):
    imageA = binarize(imageA)
    imageB = binarize(imageB)
    try:
        imageA = remove_black_background(imageA)
        imageA = cv2.resize(imageA,imageB.shape[:-1],cv2.INTER_AREA)
    except:
        pass
    imageA = imageA/255
    imageB = imageB/255
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err = err*1000
    # print(err)
    if err > 40:
        return True
    return False
def compare(direc, imageB):
    for image in glob.glob(direc+"/*.jpg"):
        imageA = cv2.imread(image)
        flag = compare_util(imageA, imageB)
        if flag == False:
            return False
    return True
        
# img2  = r'D:\projects\personal_projects\slidemaker of online learning videos\rbr\rbr_22.jpg'

# print(compare(im,im2))

# im = cv2.imread("rbr_0.jpg")
# im2 = cv2.imread("rbr_19.jpg")

# print(compare(im, im2))
