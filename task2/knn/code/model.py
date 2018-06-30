import os
import numpy as np
from keras import backend
from keras.models import Model, Sequential
from keras.layers import *
import keras.backend as K
from keras import metrics, regularizers
from keras import losses
from keras.optimizers import *
from keras.losses import *
#from keras.utils.vis_utils import plot_model

filters = 32 
l2_reg = regularizers.l2(1e-3)

def cnn_model_build(x):
    print(K.int_shape(x))
    cnn = Sequential([
            Conv2D(filters, (3, 3), padding='same', input_shape=K.int_shape(x)[1:], kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block1_conv1'),
            Activation('elu'),
            Conv2D(filters, (3, 3), kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block1_conv2'),
            Activation('elu'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool1'),
            Dropout(0.25),

            Conv2D(filters*2, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block2_conv1'),
            Activation('elu'),
            Conv2D(filters*2, (3, 3), kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block2_conv2'),
            BatchNormalization(),
            Activation('elu'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool2'),
            Dropout(0.25),

            Conv2D(filters*4, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block3_conv1'),
            Activation('elu'),
            Conv2D(filters*4, (3, 3), kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block3_conv2'),
            BatchNormalization(),
            Activation('elu'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool3'),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='elu', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'dense1'),
            Dropout(0.5)
        ])

    return cnn

def Recognition(height=32, width=32, channel=3, class_num=80):
    
    img_input = Input(shape=(height, width, channel))
    cnn = cnn_model_build(img_input)

    x = cnn(img_input)
    x = Dense(class_num, activation='softmax', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'recognition_output')(x)

    model = Model(inputs=img_input, outputs=x)
    cnn_model = Model(inputs=img_input, outputs=cnn(img_input))
    #opt = Adam(1e-4, beta_1=0.5)
    opt = Adam(1e-4)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
    print(model.summary())

    return model, cnn_model

if __name__ == '__main__':
    Recognition()
