import cv2
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# in_jpg = "./photo_select/miona/miona84.jpg"
# out_jpg = "./photo_select_out/miona_out/miona84.jpg"

var = "miona"
for i in range(1, 27):
    #入力ファイルのパスを指定
    if i < 10:
        in_jpg = "./photo_select/" + var + '/' + var + '00' + str(i) + '.jpg'
        out_jpg = "./photo_select_out64/" + var + '_out' + '/'
    else:
        in_jpg = "./photo_select/" + var + '/' + var + '0' + str(i) + '.jpg'
        out_jpg = "./photo_select_out64/" + var + '_out' + '/'
    #入力画像の表示
    # plt.show(plt.imshow(np.asarray(Image.open(in_jpg))))

    # 画像の読み込み
    print(in_jpg)
    image_gs = cv2.imread(in_jpg)

    # グレースケールに変換
    #image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
    cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    # 顔認識の実行
    face_list = cascade.detectMultiScale(image_gs,scaleFactor=1.1,minNeighbors=1,minSize=(1,1))



    #顔が１つ以上検出された時
    if len(face_list) > 0:
        no = 1;
        for rect in face_list:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image_gs[y:y + height, x:x + width]
            image = cv2.resize(dst,(64,64))
            save_path = out_jpg + '/' + 'out_('  + str(i) +')' + str(no) + '.jpg'
            cv2.imwrite(save_path, image)
            no += 1

        # for rect in face_list:
        #     image = image_gs[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

    #顔が検出されなかった時
    else:
        print("no face")



    #出力画像の表示
    # plt.show(plt.imshow(np.asarray(Image.open(out_jpg))))
