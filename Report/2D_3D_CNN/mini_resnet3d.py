"""
3D CNN based on the ResNet architecture
"""

from keras.models import Model
from keras.layers import (Dense, Conv3D, Input, Activation, Dropout,
                          Add, GlobalAveragePooling3D,
                          BatchNormalization)
from keras.activations import relu


def mini_resnet3d(input_shape, activation=relu, scalars=False, num_classes=2):
    nfeat = 16
    dil_rate = (1,1,1)
    activ = activation
    drop_rate = 0.0
    activ_end = 'softmax'
    init = 'normal'
    input1 = Input(shape=input_shape, name='input')
    
    conv1 = Conv3D(nfeat, (3, 3, 1), kernel_initializer=init,
        padding='same', name="conv1")(input1)
    bn1 = BatchNormalization(name="bn1")(conv1)
    activ1 = Activation(activ, name="activ1")(bn1)
    
    drop1 = Dropout(drop_rate, name = "drop1")(activ1)
    
    conv2 = Conv3D(nfeat * 2, (3, 3, 1), kernel_initializer=init,
        strides = (2, 2, 1), padding='same', name="conv2")(drop1)
    bn2 = BatchNormalization(name="bn2")(conv2)
    activ2 = Activation(activ, name="activ2")(bn2)
        
    conv3 = Conv3D(nfeat * 2, (3, 3, 1), kernel_initializer=init,
        padding='same', dilation_rate=dil_rate, name="conv3")(activ2)
    bn3 = BatchNormalization(name="bn3")(conv3)
    
    conv_b1 = Conv3D(nfeat * 2, (1, 1, 1), kernel_initializer=init, 
        strides = (2, 2, 1), padding='same', name="conv_b1")(drop1)
    
    skip0 = Add(name="skip0")([bn3, conv_b1])
    
    activ3 = Activation(activ, name = "activ3")(skip0)
    drop2 = Dropout(drop_rate, name = "drop2")(activ3)
    
    conv4 = Conv3D(nfeat * 4, (3, 3, 1), kernel_initializer=init, 
        strides = (2, 2, 1), padding='same', name="conv4")(drop2)
    bn4 = BatchNormalization(name="bn4")(conv4)
    activ4 = Activation(activ, name="activ4")(bn4)
        
    conv5 = Conv3D(nfeat * 4, (3, 3, 1), kernel_initializer=init,
        padding='same', dilation_rate=dil_rate, name="conv5")(activ4)
    bn5 = BatchNormalization(name="bn5")(conv5)
        
    conv_b2 = Conv3D(nfeat * 4, (1, 1, 1), kernel_initializer=init, 
        strides = (2, 2, 1), padding='same', name="conv_b2")(drop2)
    
    skip1 = Add(name="skip1")([bn5, conv_b2])
    activ5 = Activation(activ, name="activ5")(skip1)

    gap = GlobalAveragePooling3D(name = "gap")(activ5)
    drop3 = Dropout(drop_rate, name = "drop3")(gap)

    out1 = Dense(num_classes, kernel_initializer=init,
        activation=activ_end, name="out1")(drop3)
    model = Model(inputs=input1, outputs=out1)
    
    return model
