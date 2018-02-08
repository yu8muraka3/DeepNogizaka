from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np

#分類対象のカテゴリーを選ぶ
nogi_dir = "./photo_out/"
categories = ["asuka_out","ayane_out","chiharu_out","himetan_out","hinachima_out","hinako_out","ikoma_out","ikuta_out","iori_out","junna_out","kanarin_out","karin_out","kawago_out","kawamura_out","kazumin_out","kotoko_out","maichun_out","maiyan_out","manattan_out","marika_out","maya_out","minami_out","miona_out","miria_out","misamisa_out","nanase_out","nojo_out","ranze_out","reika_out","renachi_out","sayunyan_out","sayurin_out","waka_out","yuttan_out"]

nb_classes = len(categories)

#画像サイズ指定
image_w = 224
image_h = 224
pixels = image_w * image_h * 3

#画像データを読み込み
X = []
Y = []

for idx, cat in enumerate(categories):
    #ラベルを指定
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    #画像
    image_dir = nogi_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

#学習データとテストデータを分ける
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./photo/5obj.npy", xy)

print("ok", len(Y))
