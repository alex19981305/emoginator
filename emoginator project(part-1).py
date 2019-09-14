import cv2
import numpy as np
import os

image_x,image_y=50,50

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def store_images(g_id):
    total_pics = 1200
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.Color_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255]))
        res = cv2.bitwise_and(frame, frame, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)

        kernel_square = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 10000 and frames > 50:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            pic_no += 1
            save_img = thresh[y1:y1 + h1, x1:x1 + w1]
            save_img = cv2.resize(save_img, (image_x, image_y))
            cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
            cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("capturing gesture", frame)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord("c"):
            if flag_start_capturing == False:
                flag_start_capturing == True
            else:
                flag_start_capturing == False
                frames = 0
            if flag_start_capturing == True:
                frames += 1
            if pic_no == total_pics:
                break
import numpy as np
from keras.layers import Dense,Flatten,Conv2D
from keras.layers import MaxPooling2D,Dropout
from keras.utils import np_utils,print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import keras.backend as K

data=pd.read_csv("trainfoo.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X=X[:,1:2501]
y=y[:,0]

X_train=X[0:12000,:]
X_train=X_train/255
X_test=X[12000:13201,:]
X_test=X_test/255


Y=Y.reshape(Y.shape[0],1)
Y_train=Y[0:12000,:]
Y_train=Y_train.T
Y_test=Y[12000:13201,:]
Y_test=Y_test.T

image_x=50
image_y=50
train_y=np_utils.to_categorical(Y_train)
test_y=np_utils.to_categorical(Y_test)
train_y=train_y.reshape(train_y.shape[1],train_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
X_train=X_train.reshape(X_train.shape[0],image_x,image_y,1)
X_test=X_test.reshape(X_test.shape[0],image_x,image_y,1)


def keras_model(image_x, image_y):
    num_of_classes = 12
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), input_shape=(image_x, image_y, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Conv2D(64, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    filepath = "face-rec_256.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
    callbacks_list = [checkpoint1]

    return model, callbacks_list


model,callbacks_list=keras_model(image_x,image_y)
model.fit(X_train,train_y,validation_data=(X_test,test_y),epochs=10,batch_size=64,callbacks=callbacks_list)
scores=model.evaluate(X_test,test_y,verbose=0)
print("CNN error: %.2f%%" % (100-scores[1]*100))
print_summary(model)
model.save("handEmo.h5")