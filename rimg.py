# 顔認識する対象を決定（検索ワードを入力）
#SearchName = ["Class1","Class1_def","Class2","Class2_def","Class3","Class3_def","Class4","Class4_def","Class5","Class5_def"]
# 画像の取得枚数の上限
SearchName = ["Class1","Class2","Class3"]
ImgNumber =600
# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
input_shape=(128,128,3)

# 2割をテストデータに移行
import shutil
import random
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam

#for name in SearchName:
#    in_dir = "./190402_Ed/"+name+"/*"
#    in_jpg=glob.glob(in_dir)
#    img_file_name_list=os.listdir("./190402_Ed/"+name+"/")
#    #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
#    random.shuffle(in_jpg)
#    os.makedirs('./test04/' + name, exist_ok=True)
#    for t in range(len(in_jpg)//5):
#       shutil.move(str(in_jpg[t]), "./test04/"+name)

from keras.utils.np_utils import to_categorical

# 教師データのラベル付け
X_train = [] 
Y_train = [] 
for i in range(len(SearchName)):
    img_file_name_list=os.listdir("./190402_Ed/"+SearchName[i])
    print("{}:トレーニング用の写真の数は{}枚です。".format(SearchName[i],len(img_file_name_list)))

    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./190402_Ed/"+SearchName[i]+"/",img_file_name_list[j])  
        img = cv2.imread(n)
        if img is None:
            print('image' + str(j) + ':NoImage')
            continue    
        else:
            r,g,b = cv2.split(img)
            img = cv2.merge([r,g,b])
            X_train.append(img)
            Y_train.append(i)

print("")

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル（名前）
for i in range(len(SearchName)):
    img_file_name_list=os.listdir("./test04/"+SearchName[i])
    print("{}:テスト用の写真の数は{}枚です。".format(SearchName[i],len(img_file_name_list)))
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join("./test04/"+SearchName[i]+"/",img_file_name_list[j])
        img = cv2.imread(n)
        if img is None:
            print('image' + str(j) + ':NoImage')
            continue    
        else:
            r,g,b = cv2.split(img)
            img = cv2.merge([r,g,b])
            X_test.append(img)
            Y_test.append(i)

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
######################################################
# モデルの定義

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=16,kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(3, 3),
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
# 分類したい人数を入れる
model.add(Dense(len(SearchName)))
model.add(Activation('softmax'))
model.summary()
"""

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(SearchName)))
model.add(Activation('softmax'))
"""
model.summary()







# コンパイル
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



def valid(X_test,y_test):
    pred = model.predict(X_test)
    lpred=len(y_test)
    argp=np.zeros(lpred)
    argv=np.zeros(lpred)
    for i in range(0,lpred):
        argp[i]=np.argmax(pred[i,:])
        argv[i]=np.argmax(y_test[i,:])
        print('test set:{:}     predicted:{:}'.format(argv[i],argp[i]))
    predict_merge=np.c_[argv,argp,y_test, pred]
    df=pd.DataFrame(predict_merge)
    df.to_csv("result_valid.csv", sep = " ", index=None, header=None)

history = model.fit(X_train, y_train, batch_size=256,
                    epochs=50, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=32
        , verbose=0)
valid(X_test,y_test)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

#モデルを保存
model.save("MyModel.h5")
