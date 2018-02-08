# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import permutation

import os, glob, cv2, math, sys
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils

# seed値
np.random.seed(1)

# 使用する画像サイズ
img_rows, img_cols = 224, 224

# 画像データ 1枚の読み込みとリサイズを行う
def get_im(path):

    img = cv2.imread(path)
    resized = cv2.resize(img, (img_cols, img_rows))

    return resized


# データの読み込み、正規化、シャッフルを行う
def read_train_data(ho=0, kind='train'):

    train_data = []
    train_target = []

    # 学習用データ読み込み
    for j in range(0, ): # 0～5まで

        path = './photo_out'
        path += '%s/%i/*/%i/*.jpg'%(kind, ho, j)

        files = sorted(glob.glob(path))

        for fl in files:

            flbase = os.path.basename(fl)

            # 画像 1枚 読み込み
            img = get_im(fl)
            img = np.array(img, dtype=np.float32)

            # 正規化(GCN)実行
            img -= np.mean(img)
            img /= np.std(img)

            train_data.append(img)
            train_target.append(j)

    # 読み込んだデータを numpy の array に変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.uint8)

    # (レコード数,縦,横,channel数) を (レコード数,channel数,縦,横) に変換
    train_data = train_data.transpose((0, 3, 1, 2))

    # target を 6次元のデータに変換。
    # ex) 1 -> 0,1,0,0,0,0   2 -> 0,0,1,0,0,0
    train_target = np_utils.to_categorical(train_target, 6)

    # データをシャッフル
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]

    return train_data, train_target


# テストデータ読み込み
def load_test(test_class, aug_i):

    path = '../../data/Caltech-101/test/%i/%i/*.jpg'%(aug_i, test_class)

    files = sorted(glob.glob(path))
    X_test = []
    X_test_id = []

    for fl in files:
        flbase = os.path.basename(fl)

        img = get_im(fl)
        img = np.array(img, dtype=np.float32)

        # 正規化(GCN)実行
        img -= np.mean(img)
        img /= np.std(img)

        X_test.append(img)
        X_test_id.append(flbase)

    # 読み込んだデータを numpy の array に変換
    test_data = np.array(X_test, dtype=np.float32)

    # (レコード数,縦,横,channel数) を (レコード数,channel数,縦,横) に変換
    test_data = test_data.transpose((0, 3, 1, 2))

    return test_data, X_test_id


# 9層 CNNモデル 作成
def layer_9_model():

    # KerasのSequentialをモデルの元として使用 ---①
    model = Sequential()

    # 畳み込み層(Convolution)をモデルに追加 ---②
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear',
     input_shape=(3, img_rows, img_cols)))
    model.add(LeakyReLU(alpha=0.3))

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='linear'))
    model.add(LeakyReLU(alpha=0.3))

    # プーリング層(MaxPooling)をモデルに追加 ---③
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten層をモデルに追加 -- ④
    model.add(Flatten())
    # 全接続層(Dense)をモデルに追加 --- ⑤
    model.add(Dense(1024, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    # Dropout層をモデルに追加 --- ⑥
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='linear'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))
    # 最終的なアウトプットを作成。 --- ⑦
    model.add(Dense(6, activation='softmax'))

    # ロス計算や勾配計算に使用する式を定義する。 --- ⑧
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
             loss='categorical_crossentropy', metrics=["accuracy"])
    return model


# モデルの構成と重みを読み込む
def read_model(ho, modelStr='', epoch='00'):
    # モデル構成のファイル名
    json_name = 'architecture_%s_%i.json'%(modelStr, ho)
    # モデル重みのファイル名
    weight_name = 'model_weights_%s_%i_%s.h5'%(modelStr, ho, epoch)

    # モデルの構成を読込み、jsonからモデルオブジェクトへ変換
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    # モデルオブジェクトへ重みを読み込む
    model.load_weights(os.path.join('cache', weight_name))

    return model


# モデルの構成を保存
def save_model(model, ho, modelStr=''):
    # モデルオブジェクトをjson形式に変換
    json_string = model.to_json()
    # カレントディレクトリにcacheディレクトリがなければ作成
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    # モデルの構成を保存するためのファイル名
    json_name = 'architecture_%s_%i.json'%(modelStr, ho)
    # モデル構成を保存
    open(os.path.join('cache', json_name), 'w').write(json_string)


def run_train(modelStr=''):

    # HoldOut 2回行う
    for ho in range(2):

        # モデルの作成
        model = layer_9_model()

        # trainデータ読み込み
        t_data, t_target = read_train_data(ho, 'train')
        v_data, v_target = read_train_data(ho, 'valid')

        # CheckPointを設定。エポック毎にweightsを保存する。
        cp = ModelCheckpoint('./cache/model_weights_%s_%i_{epoch:02d}.h5'%(modelStr, ho),
        monitor='val_loss', save_best_only=False)

        # train実行
        model.fit(t_data, t_target, batch_size=64,
                  nb_epoch=40,
                  verbose=1,
                  validation_data=(v_data, v_target),
                  shuffle=True,
                  callbacks=[cp])


        # モデルの構成を保存
        save_model(model, ho, modelStr)



# テストデータのクラスを推測
def run_test(modelStr, epoch1, epoch2):

    # クラス名取得
    columns = []
    for line in open("../../data/Caltech-101/label.csv", 'r'):
        sp = line.split(',')
        for column in sp:
            columns.append(column.split(":")[1])

    # テストデータが各クラスに分かれているので、
    # 1クラスずつ読み込んで推測を行う。
    for test_class in range(0, 6):

        yfull_test = []

        # データ拡張した画像を読み込むために5回繰り返す
        for aug_i in range(0,5):

            # テストデータを読み込む
            test_data, test_id = load_test(test_class, aug_i)

            # HoldOut 2回繰り返す
            for ho in range(2):

                if ho == 0:
                    epoch_n = epoch1
                else:
                    epoch_n = epoch2

                # 学習済みモデルの読み込み
                model = read_model(ho, modelStr, epoch_n)

                # 推測の実行
                test_p = model.predict(test_data, batch_size=128, verbose=1)

                yfull_test.append(test_p)

        # 推測結果の平均化
        test_res = np.array(yfull_test[0])
        for i in range(1,10):
            test_res += np.array(yfull_test[i])
        test_res /= 10

        # 推測結果とクラス名、画像名を合わせる
        result1 = pd.DataFrame(test_res, columns=columns)
        result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

        # 順番入れ替え
        result1 = result1.ix[:,[6, 0, 1, 2, 3, 4, 5]]

        if not os.path.isdir('subm'):
            os.mkdir('subm')
        sub_file = './subm/result_%s_%i.csv'%(modelStr, test_class)

        # 最終推測結果を出力する
        result1.to_csv(sub_file, index=False)

        # 推測の精度を測定する。
        # 一番大きい値が入っているカラムがtest_classであるレコードを探す
        one_column = np.where(np.argmax(test_res, axis=1)==test_class)
        print ("正解数　　" + str(len(one_column[0])))
        print ("不正解数　" + str(test_res.shape[0] - len(one_column[0])))




# 実行した際に呼ばれる
if __name__ == '__main__':

    # 引数を取得
    # [1] = train or test
    # [2] = test時のみ、使用Epoch数 1
    # [3] = test時のみ、使用Epoch数 2
    param = sys.argv

    if len(param) < 2:
        sys.exit ("Usage: python 9_Layer_CNN.py [train, test] [1] [2]")

    # train or test
    run_type = param[1]

    if run_type == 'train':
        run_train('9_Layer_CNN')
    elif run_type == 'test':
        # testの場合、使用するエポック数を引数から取得する
        if len(param) == 4:
            epoch1 = "%02d"%(int(param[2])-1)
            epoch2 = "%02d"%(int(param[3])-1)
            run_test('9_Layer_CNN', epoch1, epoch2)
        else:
            sys.exit ("Usage: python 9_Layer_CNN.py [train, test] [1] [2]")
    else:
        sys.exit ("Usage: python 9_Layer_CNN.py [train, test] [1] [2]")
