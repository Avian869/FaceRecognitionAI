#import all required library 
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from tensorflow.keras.layers import (
    Input,
    Conv2D, 
    MaxPool2D, 
    Dense, 
    BatchNormalization, 
    ReLU, 
    Dropout, 
    Flatten,
    Dropout,
    Concatenate,
    GlobalAvgPool2D
)

from tensorflow.keras.regularizers import L2
import tensorflow as tf

def inception_module(inputs,f1,f2):
    # define convolution  2d layer
    x1=Conv2D(f1,3,padding="same")(inputs)
    x1=BatchNormalization()(x1)
    x1=ReLU()(x1)
    
    x2=Conv2D(f2,5,padding="same")(inputs)
    x2=BatchNormalization()(x2)
    x2=ReLU()(x2)
    
    #combine x1 and x2 
    return Concatenate()([x1,x2])

def build_model():
    #            image size 
    inputs=Input((96,96,3))
    # apply inception layer
    x=inception_module(inputs,64,32)
    # apply Max pool layer
    x=MaxPool2D()(x)
    # change input and f1,f2
    x=inception_module(x,64,32)
    x=MaxPool2D()(x)
    x=inception_module(x,128,32)
    x=MaxPool2D()(x)
    x=inception_module(x,128,32)
    x=MaxPool2D()(x)
    x=inception_module(x,256,64)
    x=MaxPool2D()(x)
    
    # apply Flatten 
    x=Flatten()(x)
    # now we will divide it for two group age and gender
    
    # for gender
    dense_g=Dense(64,activation="relu")(x)
    # apply dropout to improve model from overfitting
    drop_g=Dropout(0.3)(dense_g)
    output_g=Dense(1,activation="sigmoid",name="gender")(drop_g)
    # you can change Dense input, dropout or increase number of similar layer to
    # improve gender accuracy
    
    # for age
    x=Dense(1024,kernel_regularizer=L2(l2=0.05))(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    x=Dense(512,kernel_regularizer=L2(l2=0.02))(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    x=Dense(128,kernel_regularizer=L2(l2=0.01))(x)
    x=BatchNormalization()(x)
    x=ReLU()(x)
    
    x=Dense(1)(x)
    
    # define model input and output
    model=tf.keras.Model(inputs=[inputs],outputs=[output_g,x])
    return model

files=os.listdir("Training/UTKFace/")
files[:5]

#create an empty array to store image, labels
age_array=[]
gender_array=[]  # will convert list to array 
file_path=[]
file_name=[]
labels=[]
image_array=[]
path="Training/UTKFace/"

#loop through each files
for i in tqdm(range(len(files))):
    age=int(files[i].split("_")[0])
    # if we split filename with "_"
    # 0 position will be age
    # 1 positio will be gender
    gender=int(files[i].split("_")[1])
    # 0 : Male
    # 1: Female
    if(age<=100):
        # remove dateset for age group above 100
        # If you want me to train for imbalance class comment below(above 100)
        age_array.append(age)
        gender_array.append(gender)
        file_path.append(path+files[i])
        labels.append([[age],[gender]])
        # read image 
        image=cv2.imread(path+files[i])
        # resize image (96,96) 
        # original size is (200,200)
        # for training on (200,200) we required more ram memory 
        # so we go with (96,96)
        # resize image
        image=cv2.resize(image,(96,96))
        # conver image from BGR to RGB
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_array.append(image)
        
        # it will take time 

a,b=np.unique(age_array,return_counts="True")

a
b

image_array=np.array(image_array)/255.0
labels=np.array(labels)

from sklearn.model_selection import train_test_split
image_array,X_test,Y_train,Y_test=train_test_split(image_array,labels,test_size=0.2)

Y_train_split=[Y_train[:,1],Y_train[:,0]]
Y_test_split=[Y_test[:,1],Y_test[:,0]]

Y_train_split

model=build_model()
model.summary()

model.compile(optimizer="adam",loss=["binary_crossentropy","mean_squared_error"],metrics=["mae"])

ckp_path="Training"
model_checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path,
                                                   monitor="val_dense_4_mae",
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode="auto")

reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(factor=0.9,
                                              monitor="val_dense_4_mae",
                                              mode="auto",
                                              cooldown=0,
                                              patience=5,
                                              varbose=1,
                                              min_lr=1e-5)

Y_test_split

EPOCHS=100
BATCH_SIZE=256
history=model.fit(image_array,Y_train_split,
                 validation_data=(X_test,Y_test_split),
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 callbacks=[model_checkpoint,reduce_lr])

model.load_weights("Training")

prediction_val=model.predict(X_test,batch_size=BATCH_SIZE)

prediction_val[0][:20]
Y_test_split[0][:20]
prediction_val[1][:20]
Y_test_split[1][:20]

# convert model into tensorflow lite model
converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

#save model 
with open("model.tflite","wb") as f:
    f.write(tflite_model)