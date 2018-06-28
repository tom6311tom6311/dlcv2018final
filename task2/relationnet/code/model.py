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

filters = 64

def RelationNet(height=32, width=32, channel=3):
    l2_reg = regularizers.l2(5e-4)

    img_input1 = Input(shape=(height, width, channel))
    img_input2 = Input(shape=(height, width, channel))

    cnn = Sequential([
            Conv2D(filters, (3, 3), padding='same', input_shape=(height, width, channel), kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block1_conv'),
            #BatchNormalization(),
            #Activation('relu'),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool1'),

            Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block2_conv'),
            BatchNormalization(),
            #Activation('relu'),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool2'),

            Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block3_conv'),
            BatchNormalization(),
            #Activation('relu'),
            Activation('tanh'),

            Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block4_conv'),
            BatchNormalization(),
            #Activation('relu'),
            Activation('tanh'),

            #MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool4')

        ])

    cnn_feature1 = cnn(img_input1)
    cnn_feature2 = cnn(img_input2)

    x1 = Concatenate()([cnn_feature1, cnn_feature2])
    x2 = Concatenate()([cnn_feature2, cnn_feature1])
    _, h, w, c = Model(inputs=[img_input1, img_input2], outputs=x1).output_shape

    rnet = Sequential([
            Conv2D(filters, (5, 5), padding='same', input_shape=(h, w, c), kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block5_conv'),
            BatchNormalization(),
            #Activation('relu'),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool5'),

            Conv2D(filters*2, (5, 5), padding='same', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name='block6_conv'),
            BatchNormalization(),
            #Activation('relu'),
            Activation('tanh'),

            MaxPooling2D((2, 2), strides=(2, 2), padding='same', name = 'pool6'),


            Flatten(input_shape = (h, w, c)),
            #Dense(1024, activation='relu', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'dense1'),
            Dense(512, activation='tanh', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'dense1'),
            Dropout(0.5),

            Dense(4, activation='tanh', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'dense2'),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2_reg, bias_regularizer=l2_reg, name = 'output')
        ])

    
    logits1 = rnet(x1)
    logits2 = rnet(x2)
    logits = Average()([logits1, logits2])

    f_input1 = Input(shape=(h, w, int(c/2)))
    f_input2 = Input(shape=(h, w, int(c/2)))
    x = Concatenate()([f_input1, f_input2])
    f_logits = rnet(x)

    def total_loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) + 1e-1 * losses.mean_squared_error(logits1, logits2)

    model = Model(inputs=[img_input1, img_input2], outputs=logits)
    cnn_model = Model(img_input1, cnn_feature1)
    relation_model = Model(inputs=[f_input1, f_input2], outputs=f_logits)
    #opt = Adam(1e-4, beta_1=0.5)
    opt = Adam(1e-4)
    #model.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    model.compile(loss=total_loss, optimizer=opt, metrics=['acc'])

    return model, cnn_model, relation_model


if __name__ == '__main__':
    RelationNet()
