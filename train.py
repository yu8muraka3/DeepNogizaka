from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.optimizers import Adam



#分類対象のカテゴリーを選ぶ
nogi_dir = "./photo_select_out64/"
categories = ["asuka_out", "ikoma_out", "ikuta_out", "maiyan_out", "nanase_out", "yasushi_out"]

nb_classes = len(categories)

#画像サイズ指定
image_w = 64
image_h = 64

#画像データを読み込み
X = []
Y = []

X = np.array(X)
Y = np.array(Y)


X_train, X_test, y_train, y_test = np.load("./5obj.npy")

X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# モデルの定義
model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
    optimizer=Adam(lr=1e-5),
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=15, validation_split=0.1)

score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


#モデルを保存
model.save("my_model.h5")
