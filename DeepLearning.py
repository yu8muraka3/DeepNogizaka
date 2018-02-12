from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

#カテゴリーの指定
categories = ["asuka_out","ayane_out","chiharu_out","himetan_out","hinachima_out","hinako_out","ikoma_out","ikuta_out","iori_out","junna_out","kanarin_out","karin_out","kawago_out","kawamura_out","kazumin_out","kotoko_out","maichun_out","maiyan_out","manattan_out","marika_out","maya_out","minami_out","miona_out","miria_out","misamisa_out","nanase_out","nojo_out","ranze_out","reika_out","renachi_out","sayunyan_out","sayurin_out","waka_out","yuttan_out"]

nb_classes = len(categories)

image_w = 224
image_h = 224

X_train, X_test, y_train, y_test = np.load("5obj.npy", encoding = "latin1")

X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
X_train = X_train.transpose((0, 3, 1, 2))
X_test = X_test.transpose((0,3,1,2))
print('X_train shape:', X_train.shape)

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

model.fit(X_train, y_train, batch_size=64, nb_epoch=10)

score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])
