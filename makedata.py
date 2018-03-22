from PIL import Image
import os, glob
import numpy as np
import random, math

root_dir = "./photo_select_out64/"
categories = ["asuka_out", "ikoma_out", "ikuta_out", "maiyan_out", "miona_out", "nanase_out", "yasushi_out"]
nb_classes = len(categories)
image_size = 64

# 画像データを読み込む
X = []  # 画像データ
Y = []  # ラベルデータ

def add_sample(cat, fname, is_train):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)
    if not is_train: return

    for ang in range(-20, 20, 5):
        img2 = img.rotate(ang)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)

        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)

def make_sample(files, is_train):
    global X, Y
    X = []; Y = []
    for cat, fname in files:
        add_sample(cat, fname, is_train)
    return np.array(X), np.array(Y)

allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.6)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train, True)
X_test, y_test = make_sample(test, False)
xy = (X_train, X_test, y_train, y_test)
np.save("./5obj.npy", xy)
print("ok", len(y_train))