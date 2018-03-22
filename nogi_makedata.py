from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils


#分類対象のカテゴリーを選ぶ
nogi_dir = "./face_scratch_image/"
categories = ["asuka", "ikoma", "ikuta", "maiyan", "miona", "nanase", "yasushi"]

nb_classes = len(categories)

#画像サイズ指定
image_w = 224
image_h = 224

#画像データを読み込み
X = []
Y = []

for idx, cat in enumerate(categories):
    #ラベルを指定
    # label = [0 for i in range(nb_classes)]
    # label[idx] = 1
    #画像
    image_dir = nogi_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)


#学習データとテストデータを分ける
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./5obj.npy", xy)

print("ok", len(Y))

X_train, X_test, y_train, y_test = np.load("./5obj.npy")

X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# モデルの定義
model = Sequential()
model.add(Conv2D(input_shape=(224, 224, 3), filters=32,kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=20)

score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


#モデルを保存
model.save("my_model.h5")
