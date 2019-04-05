from keras.models import Model
from keras.layers import (Dense, Conv2D, Input, Activation, Dropout,
                          concatenate, MaxPooling2D,
                          GlobalMaxPooling2D, BatchNormalization)
from keras.activations import relu


def network_3Dcnn(input_shape, activation=relu, num_classes=2):
    """
    3D CNN
    """
    nfeat = 32
    dropout_rate = 0.3
    dil_rate = (2, 2, 1)
    batch_normalization = False
    activ = activation
    activ_end = 'softmax'
    init = 'normal'
    input1 = Input(shape=input_shape, name='input')

    if not batch_normalization:

        conv1 = Conv3D(nfeat, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate = dil_rate,
                       name="conv1")(input1)
        activ1 = Activation(activ, name="activ1")(conv1)

        conv2 = Conv3D(nfeat, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv2")(activ1)
        activ2 = Activation(activ, name="activ2")(conv2)
        
        skip0 = concatenate([activ1, activ2],axis=-1, name = "skip0")
        
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name="pool1")(skip0)
        drop1 = Dropout(dropout_rate, name="drop1")(pool1)

        conv5 = Conv3D(nfeat * 2, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv5")(drop1)
        activ5 = Activation(activ, name="activ5")(conv5)

        conv6 = Conv3D(nfeat * 2, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv6")(activ5)
        activ6 = Activation(activ, name="activ6")(conv6)
        
        skip1 = concatenate([activ6, pool1],axis=-1, name = "skip1")
        
        gmp = GlobalMaxPooling3D(name="gmp")(skip1)
        drop2 = Dropout(dropout_rate, name="drop2")(gmp)

        dense1 = Dense(50, activation=activ, name="dense1")(drop2)
        activ7 = Activation(activ, name="activ7")(dense1)
        drop3 = Dropout(dropout_rate, name="drop3")(activ7)

    else:
        conv1 = Conv3D(nfeat, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv1")(input1)
        bn1 = BatchNormalization(name="bn1")(conv1)
        activ1 = Activation(activ, name="activ1")(bn1)

        conv2 = Conv3D(nfeat, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv2")(activ1)
        bn2 = BatchNormalization(name="bn2")(conv2)
        activ2 = Activation(activ, name="activ2")(bn2)

        skip0 = concatenate([activ1, activ2],axis=-1, name = "skip0")
        pool1 = MaxPooling3D(pool_size=(2, 2, 2), name="pool1")(skip0)
        drop1 = Dropout(dropout_rate, name="drop1")(pool1)

        conv5 = Conv3D(nfeat * 2, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv5")(drop1)
        bn5 = BatchNormalization(name="bn5")(conv5)
        activ5 = Activation(activ, name="activ5")(bn5)

        conv6 = Conv3D(nfeat * 2, (3, 3, 3), kernel_initializer=init,
                       padding='same', dilation_rate=dil_rate,
                       name="conv6")(activ5)
        bn6 = BatchNormalization(name="bn6")(conv6)
        activ6 = Activation(activ, name="activ6")(bn6)

        skip1 = concatenate([pool1, activ6],axis=-1, name = "skip1")
        gmp = GlobalMaxPooling3D(name="gmp")(skip1)
        drop2 = Dropout(dropout_rate, name="drop2")(gmp)
        
        dense1 = Dense(50, activation=activ, name="dense")(drop2)
        activ7 = Activation(activ, name="activ7")(dense1)
        drop3 = Dropout(dropout_rate, name="drop3")(activ7)

    out1 = Dense(num_classes, kernel_initializer=init,
        activation=activ_end, name="out1")(drop3)
    model = Model(inputs=input1, outputs=out1)
   
    return model
