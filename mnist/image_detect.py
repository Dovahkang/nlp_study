# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:59:15 2017

@author: jhak
"""
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2
from PIL import Image
import sys

path = '/home/dovah/CreativeBanner/image/'
img0 = Image.open(path + 'POC_UA/hive/1.jpg').convert('RGB')

banner = np.array(img0)
banner = cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)
#banner = cv2.imread(path+'/POC_배경/영지전체이미지_건물없음.jpg')

img1 = Image.open(path + '/POC_BI/POC_ENG_01.jpg').convert('RGB')
template = np.array(img1)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

newx, newy = template.shape[1] / 22, template.shape[0] / 22  # new size (w,h)
newimage = cv2.resize(template, (int(newx), int(newy)))

newimage = cv2.Canny(newimage, 50, 200)
(tempheight, tempwidth) = newimage.shape[::-1]

banner_canny = cv2.Canny(banner, newx, newy)
res = cv2.matchTemplate(banner_canny, newimage, cv2.TM_CCOEFF_NORMED)
threshold = np.amax(res)

loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(banner, pt, (pt[0]+tempheight, pt[1]+ tempwidth), (0, 255, 0), 1)
    print(pt)

img_ch = Image.open(path + '/POC_Character/jpg/Jack Sparrow_version03_completion.jpg').convert('RGB')
template_ch = np.array(img_ch)
template_ch = cv2.cvtColor(template_ch, cv2.COLOR_BGR2GRAY)
new_temp_ch = cv2.Canny(template_ch, 50, 200)

newx, newy = template_ch.shape[1] / 1.5, template_ch.shape[0] / 2  # new size (w,h)
new_temp_ch = cv2.resize(new_temp_ch, (int(newx), int(newy)))
(tempheight, tempwidth) = new_temp_ch.shape[::-1]

res = cv2.matchTemplate(banner_canny, new_temp_ch, cv2.TM_CCOEFF_NORMED)

threshold = np.amax(res)

loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(banner, pt, (pt[0]+tempheight, pt[1]+ tempwidth), (0, 255, 0), 1)
    print(pt)
cv2.imshow('Detect resource image', banner)
cv2.imshow('Canny banner image', newimage)
cv2.waitKey(0)