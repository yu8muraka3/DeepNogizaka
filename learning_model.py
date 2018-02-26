import random
from keras.utils.np_utils import to_categorical
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


img_file_name_list=os.listdir("./face_scratch_image/")
print(len(img_file_name_list))

for i in range(len(img_file_name_list)):
    n=os.path.join("./face_scratch_image",img_file_name_list[i])
    img = cv2.imread(n)
    if isinstance(img,type(None)) == True:
        img_file_name_list.pop(i)
        continue
print(len(img_file_name_list))

X_train=[]
y_train=[]

for j in range(0,len(img_file_name_list)-1):
    n=os.path.join("./face_scratch_image/",img_file_name_list[j])
    img = cv2.imread(n)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    X_train.append(img)
    n=img_file_name_list[j]
    y_train=np.append(y_train,int(n[0:2])).reshape(j+1,1)

X_train=np.array(X_train)

img_file_name_list=os.listdir("./test_image1/")
print(len(img_file_name_list))

for i in range(len(img_file_name_list)):
    n=os.path.join("./test_image1",img_file_name_list[i])
    img = cv2.imread(n)
    if isinstance(img,type(None)) == True:
        img_file_name_list.pop(i)
        continue
print(len(img_file_name_list))

X_test=[]
y_test=[]

for j in range(0,len(img_file_name_list)):
    n=os.path.join("./test_image1",img_file_name_list[j])
    img = cv2.imread(n)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    X_test.append(img)
    n=img_file_name_list[j]
    y_test=np.append(y_test,int(n[0:2])).reshape(j+1,1)

X_test=np.array(X_test)
