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
from keras.metrics import *
#from keras.utils.vis_utils import plot_model

filters = 64 
l2_reg = regularizers.l2(5e-4)

def cnn_model_build(x):
    cnn = Sequential([
            Conv2D(filters, (3, 3), padding='same', input_shape=K.int_shape(x)[1:], 
                kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block1_conv1'),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool1'),
            #Dropout(0.5),

            Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block2_conv1'),
            BatchNormalization(),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool2'),
            Dropout(0.25),

            Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block3_conv1'),
            BatchNormalization(),
            Activation('tanh'),

            Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block3_conv2'),
            BatchNormalization(),
            Activation('tanh'),

        ])
    return cnn

def cnn2_model_build(x):
    cnn = Sequential([
            Conv2D(filters, (5, 5), input_shape=K.int_shape(x)[1:], padding='same',
                kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block4_conv1'),
            BatchNormalization(),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool4'),
            #Dropout(0.5),

            Conv2D(filters*2, (5, 5), padding='same',
                kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block5_conv1'),
            BatchNormalization(),
            #Activation('elu'),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool5'),
            Dropout(0.5),

            Flatten(),
            Dense(512, activation='tanh', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'dense1'),
            Dropout(0.5)
        ])

    return cnn

def rnet_model_build(x):
    rnet = Sequential([
            Dense(4, activation='tanh', input_shape=K.int_shape(x)[1:], kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'dense2'),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'output')
        ])
    return rnet

def RelationNet(height=32, width=32, channel=3):

    img_input1 = Input(shape=(height, width, channel))
    img_input2 = Input(shape=(height, width, channel))

    cnn = cnn_model_build(img_input1)

    cnn_feature1 = cnn(img_input1)
    cnn_feature2 = cnn(img_input2)

    x1 = Concatenate()([cnn_feature1, cnn_feature2])
    x2 = Concatenate()([cnn_feature2, cnn_feature1])

    cnn2 = cnn2_model_build(x1)
    rnet = rnet_model_build(cnn2(x1))

    logits1 = rnet(cnn2(x1))
    logits2 = rnet(cnn2(x2))
    logits = Average()([logits1, logits2])

    f_input1 = Input(shape=K.int_shape(cnn_feature1)[1:])
    f_input2 = Input(shape=K.int_shape(cnn_feature1)[1:])
    x = Concatenate()([f_input1, f_input2])
    f_logits = rnet(cnn2(x))

    def total_loss(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) + 5e-1* mean_squared_error(logits1, logits2) #+ -5e-1 * (y_true*2-1) * y_pred

    def acc(y_trud, y_pred):
        return binary_accuracy(y_trud, y_pred)

    def loss1(y_trud, y_pred):
        return K.sqrt(mean_squared_error(logits1, logits2))

    def loss2(y_true, y_pred):
        return (y_true*2-1) * y_pred

    model = Model(inputs=[img_input1, img_input2], outputs=logits)
    cnn_model = Model(img_input1, cnn_feature1)
    relation_model = Model(inputs=[f_input1, f_input2], outputs=f_logits)
    opt = Adam(1e-4, beta_1=0.5)
    model.compile(loss=total_loss, optimizer=opt, metrics=[acc, loss1, loss2])

    #print(model.summary())

    return model, cnn_model, relation_model


