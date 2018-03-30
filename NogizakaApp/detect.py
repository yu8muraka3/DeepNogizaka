import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from flask import request
from operator import itemgetter

def detect_face(image):
    print(image.shape)
    #opencvを使って顔抽出
    # image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1,minSize=(64,64))
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        face_info = [[]]
        for rect in face_list:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            face_size = width * height
            img = image[y:y + height, x:x + width]
            face_info.append([face_size, img, rect[0:2], rect[2:4], x, y, height])
            # x,y,width,height=rect

            # img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue

        face_info.sort(reverse = True)
        predict_img = face_info[0][1]
        print(face_info[0][2:4])
        cv2.rectangle(image, tuple(face_info[0][2]), tuple(face_info[0][2]+face_info[0][3]), (255, 0, 0), thickness=3)
        img = cv2.resize(predict_img,(64, 64))
        img = np.expand_dims(img,axis=0)
        predict_name, predict_enname, rate = detect_who(img)
        show = cv2.putText(image,predict_enname,(face_info[0][4]-20,face_info[0][5]+face_info[0][6]+80),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0),2)
        cv2.imwrite("./static/asset/uploads/" + img_file, show[:, :, ::-1].copy())

        return predict_name, predict_enname, rate
    #顔が検出されなかった時
    else:
        print("no face")
        return None, None

def detect_who(img):
    #予測
    name=""
    model = load_model('./my_model.h5')

    # nameNumLabel=np.argmax(model.predict(img))
    predict = model.predict_proba(img)
    for i, pre in enumerate(predict):
        idx = np.argmax(pre)
        rate = pre[idx] * 100
        rate = str(round(rate, 1))
        if idx == 0:
            name="齋藤飛鳥"
            en_name="Saito Asuka"
        elif idx == 1:
            name="生駒里奈"
            en_name="Ikoma Rina"
        elif idx == 2:
            name="生田絵梨花"
            en_name="Ikuta Erika"
        elif idx == 3:
            name="白石麻衣"
            en_name="Shiraishi Mai"
        elif idx == 4:
            name="西野七瀬"
            en_name="Nishino Nanase"
        elif idx == 5:
            name="秋元康"
            en_name="Akimoto Yasushi"

    return name, en_name, rate

def start_detect(input_file):
    global img_file
    img_file = input_file
    model = load_model('./my_model.h5')

    image=cv2.imread("./static/asset/uploads/" + img_file)
    print(img_file)
    if image is None:
        print("Not open:")
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    predict_name, predict_enname, rate = detect_face(image)
    # whoImage=detect_face(image)
    #
    # plt.imshow(whoImage)
    # plt.show()

    return predict_name, predict_enname, rate
