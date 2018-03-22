import cv2
import numpy as np
import sys, os
from PIL import Image

in_jpg = "./photo_select_out/yasushi_out/"
out_jpg = "./photo_select_out64/yasushi_out/"

def get_file(path):
    filenames = os.listdir(path)
    return filenames

pic = get_file(in_jpg)

for i in pic:
    # 画像の読み込み
    image_path = os.path.join(in_jpg, i)
    image = cv2.imread(image_path)

    img = cv2.resize(image, (64, 64))

    a = cv2.imwrite(out_jpg + i, img)
