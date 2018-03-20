from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils


#分類対象のカテゴリーを選ぶ
nogi_dir = "./face_scratch_image"
categories = ["asuka", "ikuta", "maiyan", "miona", "nanase", "yasushi"]

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
np.save("./5obj.npy", xy)

print("ok", len(Y))

X_train, X_test, y_train, y_test = np.load("./5obj.npy")

X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
y_train = np_utils.to_categolical(y_train, nb_classes)
y_test = np_utils.to_categolical(y_test, nb_classes)


model = Sequential()

model.add(Convolution2D(32, 3, 3,
    border_mode = 'same',
    activation = 'linear',
    # input_shape = (image_w, image_h, 3)))
    input_shape = X_train.shape[1:]))
model.add(LeakyReLU(alpha=0.3))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear'))
model.add(LeakyReLU(alpha=0.3))

model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
model.add(LeakyReLU(alpha=0.3))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='linear'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=30)

score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


#モデルを保存
model.save("my_model.h5")
