import cv2
import numpy as np
import sys, os
from PIL import Image


#入力ファイルのパスを指定
in_jpg = "./photo_select/maiyan/"
out_jpg = "./photo_select_out/maiyan_out/"

#リストで結果を返す関数
def get_file(dir_path):
    filenames = os.listdir(dir_path)
    return filenames

pic = get_file(in_jpg)

for i in pic:
    # 画像の読み込み
    image_gs = cv2.imread(in_jpg + i)

    # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
    cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(1,1))

    # 顔だけ切り出して保存
    no = 1;
    for rect in face_list:
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        dst = image_gs[y:y + height, x:x + width]
        image = cv2.resize(dst,(224,224))
        save_path = out_jpg + '/' + 'out_('  + str(i) +')' + str(no) + '.jpg'

        #認識結果の保存
        a = cv2.imwrite(save_path, image)
        #plt.show(plt.imshow(np.asarray(Image.open(save_path))))
        print(no)
        no += 1
