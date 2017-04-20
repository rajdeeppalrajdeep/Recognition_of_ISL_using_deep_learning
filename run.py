
####just copy and paste the below given code to your shell

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow
import cv2
#from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 250, 250

# number of channels
img_channels = 1

#%%
#  data
path2 = 'final'  #path of folder to save images    
imlist = os.listdir(path2)

print 'final' + '//'+ imlist[0]


#temp = imlist[0].replace(" ","\ " )
#temp = temp.replace(")","\)")
#temp = temp.replace("(","\(")

#for i in range(0,len(imlist)):
#	temp = imlist[i].replace(" ","\ " )
#	temp = temp.replace(")","\)")
#	temp = temp.replace("(","\(")
#	imlist[i] = temp




im1 = array(cv2.imread('final//'+imlist[0],0)) 

imlist.sort()
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images
immatrix = array([array(cv2.imread('final//' + im2,0)).flatten()
              for im2 in imlist],'f')


num_samples = imnbr
label=np.ones((num_samples,),dtype = int)
label[0:51]=0
label[51:95]=1
label[95:140]=2
label[140:187]=3
label[187:231]=4
label[231:278]=5
label[278:323]=6
label[323:368]=7
label[368:412]=8
label[412:457]=9
label[457:501]=10
label[501:545]=11
label[545:590]=12
label[590:635]=13
label[635:679]=14
label[679:724]=15
label[724:769]=16
label[769:813]=17
label[813:861]=18
label[861:906]=19
label[906:950]=20
label[950:994]=21
label[994:1036]=22
label[1036:1081]=23
label[1081:1126]=24
label[1126:1171]=25

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 26
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3



#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



#define model
model = Sequential()
model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, img_rows, img_cols), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(15, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


#



               


