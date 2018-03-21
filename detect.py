import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model


def detect_face(image):
    print(image.shape)
    #opencvを使って顔抽出
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))
    #顔が１つ以上検出された時
    if len(face_list) > 0:
        for rect in face_list:
            x,y,width,height=rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if image.shape[0]<64:
                print("too small")
                continue
            img = cv2.resize(image,(224, 224))
            img = np.expand_dims(img,axis=0)
            name = detect_who(img)
            cv2.putText(image,name,(x,y+height+20),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
    #顔が検出されなかった時
    else:
        print("no face")
    return image

def detect_who(img):
    #予測
    name=""
    print(model.predict(img))
    nameNumLabel=np.argmax(model.predict(img))

    if nameNumLabel == 0:
        name="Saito Asuka"
    elif nameNumLabel == 1:
        name="Ikuta Erika"
    elif nameNumLabel == 2:
        name="Shiraishi Mai"
    elif nameNumLabel == 3:
        name="Hori Miona"
    elif nameNumLabel == 4:
        name="Nishino Nanase"
    elif nameNumLabel == 5:
        name="Akimoto Yasushi"
    return name

model = load_model('./my_model.h5')
print(model.summary())

image=cv2.imread("./photo_select/nanase/nanase0.jpg")
if image is None:
    print("Not open:")
b,g,r = cv2.split(image)
image = cv2.merge([r,g,b])
whoImage=detect_face(image)

plt.imshow(whoImage)
plt.show()
