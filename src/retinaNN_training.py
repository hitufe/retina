###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Conv2DTranspose,add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD

import sys
sys.path.insert(0, './lib/')
from help_functions import *

#function to obtain data for training/testing (validation)
from extract_patches import get_data_training


# Define the neural network
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)  # 上采样，行列都扩大2倍
    up1 = concatenate([conv2, up1], axis=1)  # 把这两个水平相接
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)   # 第一维度和第二维度互换
    ############
    conv7 = core.Activation('softmax')(conv6)  # softmax激活函数

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
# sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数
    return model

def get_sumnet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    sub1 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(sub1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    add1 = add([pool1, conv2])
    sub2 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(add1)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(sub2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    add2 = add([pool2, conv3])
    sub3 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(add2)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub3)  # 上采样，行列都扩大2倍
    con1 = concatenate([add1, up1], axis=1)  # 把这两个水平相接
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(con1)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    add3 = add([up1, conv4])
    sub4 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(add3)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub4)
    con2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(con2)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    add4 = add([up2, conv5])
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add4)
    conv6 = core.Reshape((2, patch_height*patch_width))(conv6)
    conv6 = core.Permute((2, 1))(conv6)   # 第一维度和第二维度互换
    ############
    conv7 = core.Activation('softmax')(conv6)  # softmax激活函数

    model = Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
# sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数
    return model

# define wnet
def get_wunet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    # 2.......................
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(conv5)
    #
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first')(conv6)
    #
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)

    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up3 = concatenate([conv6, up3], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up4 = concatenate([conv5, up4], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    # 3....................................................
    pool5 = MaxPooling2D((2, 2), data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool5)
    conv10 = Dropout(0.5)(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv10)
    pool6 = MaxPooling2D((2, 2), data_format='channels_first')(conv10)
    #
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool6)
    conv11 = Dropout(0.5)(conv11)
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv11)

    up5 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv11)
    up5 = concatenate([conv10, up5], axis=1)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up5)
    conv12 = Dropout(0.5)(conv12)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv12)
    #
    up6 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv12)
    up6 = concatenate([conv9, up6], axis=1)
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up6)
    conv13 = Dropout(0.5)(conv13)
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv13)

    conv14 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv13)
    conv14 = core.Reshape((2, patch_height*patch_width))(conv14)
    conv14 = core.Permute((2, 1))(conv14)
    ############
    conv15 = core.Activation('softmax')(conv14)

    model = Model(inputs=inputs, outputs=conv15)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
#Define the neural network gnet
#you need change function call "get_unet" to "get_gnet" in line 166 before use this network
def get_gnet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    up1 = UpSampling2D(size=(2, 2))(conv1)
    #
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool1 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool2 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool3 = MaxPooling2D((2, 2))(conv4)
    #
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv4, up2], axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv3, up3], axis=1)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    #
    up4 = UpSampling2D(size=(2, 2))(conv7)
    up4 = concatenate([conv2, up4], axis=1)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    pool4 = MaxPooling2D((2, 2))(conv8)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_uunet(n_ch,patch_height,patch_width):
    inputs = Input((n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    #
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    #
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv5 = Dropout(0.2)(conv5)
    #
    up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = concatenate([conv4, up1], axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    #
    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = concatenate([conv3, up2], axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    #
    up3 = UpSampling2D(size=(2, 2))(conv7)
    up3 = concatenate([conv2, up3], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    #
    up4 = UpSampling2D(size=(2, 2))(conv8)
    up4 = concatenate([conv1, up4], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    #
    conv10 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv10 = core.Reshape((2, patch_height * patch_width))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    ############
    conv10 = core.Activation('softmax')(conv10)

    model = Model(input=inputs, output=conv10)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# def adnet
def get_adnet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    sub1 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format='channels_first')(sub1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    add1 = add([pool1, conv2])
    sub2 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(add1)
    pool2 = MaxPooling2D((2, 2), data_format='channels_first')(sub2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    add2 = add([pool2, conv3])
    sub3 = Conv2D(256, (1, 1), activation='relu', padding='same', data_format='channels_first')(add2)
    pool3 = MaxPooling2D((2, 2), data_format='channels_first')(sub3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    add3 = add([pool3, conv4])
    sub4 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(add3)
    up1 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub4)  # 上采样，行列都扩大2倍
    con1 = concatenate([add2, up1], axis=1)  # 把这两个水平相接
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(con1)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    add4 = add([up1, conv5])
    sub5 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(add4)
    #
    up2 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub5)
    con2 = concatenate([add1, up2], axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(con2)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    add5 = add([up2, conv6])
    sub6 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(add5)
    up3 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub6)
    con3 = concatenate([conv1, up3], axis=1)
    #
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(con3)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)
    add6 = add([up3, conv7])
    # 第二轮
    sub7 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(add6)
    pool4 = MaxPooling2D((2, 2), data_format='channels_first')(sub7)
    #
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)
    add7 = add([pool4, conv8])
    sub8 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(add7)
    pool5 = MaxPooling2D((2, 2), data_format='channels_first')(sub8)
    #
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool5)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    add8 = add([pool5, conv9])
    sub9 = Conv2D(256, (1, 1), activation='relu', padding='same', data_format='channels_first')(add8)
    pool6 = MaxPooling2D((2, 2), data_format='channels_first')(sub9)
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool6)
    conv10 = Dropout(0.5)(conv10)
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv10)
    add9 = add([pool6, conv10])
    sub10 = Conv2D(128, (1, 1), activation='relu', padding='same', data_format='channels_first')(add9)
    up4 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub10)  # 上采样，行列都扩大2倍
    con4 = concatenate([add8, up4], axis=1)  # 把这两个水平相接
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(con4)
    conv11 = Dropout(0.5)(conv11)
    conv11 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv11)
    add10 = add([up4, conv11])
    sub11 = Conv2D(64, (1, 1), activation='relu', padding='same', data_format='channels_first')(add10)
    #
    up5 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub11)
    con5 = concatenate([add7, up5], axis=1)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(con5)
    conv12 = Dropout(0.5)(conv12)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv12)
    add11 = add([up5, conv12])
    sub12 = Conv2D(32, (1, 1), activation='relu', padding='same', data_format='channels_first')(add11)
    up6 = UpSampling2D(size=(2, 2), data_format='channels_first')(sub12)
    con6 = concatenate([add6, up6], axis=1)
    #
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(con6)
    conv13 = Dropout(0.5)(conv13)
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv13)
    add12 = add([up6, conv13])
    # 最后
    conv14 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(add12)
    conv14 = core.Reshape((2, patch_height*patch_width))(conv14)
    conv14 = core.Permute((2, 1))(conv14)   # 第一维度和第二维度互换
    ############
    conv15 = core.Activation('softmax')(conv14)  # softmax激活函数

    model = Model(inputs=inputs, outputs=conv15)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# sgd:随机梯度下降 categorical_crossentropy：多分类的对数损失函数,与softmax分类器相对应的损失函数
    return model
#========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('configuration.txt')
#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)


#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0], 40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'./'+name_experiment+'/'+"sample_input_masks")#.show()


#=========== Construct and save the model arcitecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]
model = get_sumnet(n_ch, patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
print(model.output_shape)
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)



#============  Training ==================================
checkpointer = ModelCheckpoint(filepath='./'+name_experiment+'/'+name_experiment +'_best_weights.h5', verbose=1, monitor='val_acc', mode='max', save_best_only=True) #save at each epoch if the validation decreased


# def step_decay(epoch):
#     lrate = 0.01 #the initial learning rate (by default in keras)
#     if epoch==100:
#         return 0.005
#     else:
#         return lrate
#
# lrate_drop = LearningRateScheduler(step_decay)

patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
# history = LossHistory()
# tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)  # 在当前目录新建logs文件夹，记录 evens.out

his = model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])

# history.loss_plot('epoch')


lossy = his.history['loss']
accy = his.history['acc']
accv = his.history['val_acc']
lossv = his.history['val_loss']
np_loss = np.array(lossy)
np_acc = np.array(accy)
np_accv = np.array(accv)
np_lossv = np.array(lossv)
np.savetxt('test/lossy.txt', np_lossv)
np.savetxt('test/loss.txt', np_loss)
np.savetxt('test/acc.txt', np_acc)
np.savetxt('test/accv.txt', np_accv)
# ========== Save and test the last model ===================
model.save_weights('./'+name_experiment+'/'+name_experiment + '_last_weights.h5', overwrite=True)
# test the model
# score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

plt.figure()
N = N_epochs
plt.plot(np.arange(0, N), his.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), his.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), his.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), his.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on SegNet Satellite Seg")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('test/plot.png')