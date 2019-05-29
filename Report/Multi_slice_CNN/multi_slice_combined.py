import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, Conv1D, 
    Input, Activation, Dropout, Concatenate, Add, 
    GlobalAveragePooling1D, GlobalAveragePooling2D,
    BatchNormalization)
from keras.activations import relu
import numpy as np


class Multi_Resnet_scalars(tf.keras.Model):
    
    def __init__(self, num_classes=2):
        super(Multi_Resnet_scalars, self).__init__()
        
        activ = relu
        activ_end = 'softmax'
        init = 'he_normal'
        drop_rate = 0.0
        nfeat = 16
        dil_rate = 1
        
        self.conv1 = Conv2D(nfeat, (3, 3), kernel_initializer=init,
            padding='same', name="conv1", 
            input_shape = (360, 360, 2))
        self.bn1 = BatchNormalization(name="bn1")
        self.activ1 = Activation(activ, name="activ1")
    
        self.drop1 = Dropout(drop_rate, name = "drop1")
        
        self.conv2 = Conv2D(nfeat * 2, (3, 3), kernel_initializer=init,
            strides = 2, padding='same', name="conv2")
        self.bn2 = BatchNormalization(name="bn2")
        self.activ2 = Activation(activ, name="activ2")

        self. conv3 = Conv2D(nfeat * 2, (3, 3), kernel_initializer=init,
            padding='same', name="conv3")
        self.bn3 = BatchNormalization(name="bn3")
    
        self.conv_b1 = Conv2D(nfeat*2, (1, 1), kernel_initializer=init, 
            strides = 2, padding='same', name="conv_b1")
    
        self.skip0 = Add(name="skip0")
        self.activ3 = Activation(activ, name = "activ3")
        self.drop2 = Dropout(drop_rate, name = "drop2")
    
        self.conv4 = Conv2D(nfeat * 4, (3, 3), kernel_initializer=init, 
            strides = 2, padding='same', name="conv4")
        self.bn4 = BatchNormalization(name="bn4")
        self.activ4 = Activation(activ, name="activ4")
        
        self.conv5 = Conv2D(nfeat * 4, (3, 3), kernel_initializer=init,
            padding='same', name="conv5")
        self.bn5 = BatchNormalization(name="bn5")
        
        self.conv_b2 = Conv2D(nfeat * 4, (1, 1), kernel_initializer=init, 
            strides = 2, padding='same', name="conv_b2")
    
        self.skip1 = Add(name="skip1")
        self.activ5 = Activation(activ, name="activ5")

        self.gap = GlobalAveragePooling2D(name = "gap")
        
        self.gap_scalars = GlobalAveragePooling3D(name = "gap_scalars")
        
        self.conc = Concatenate(axis = 0, name = "merge")
        self.conc_scalars = Concatenate(axis = 1, name = "conc_scalars")
        
        self.conv_final0 = Conv1D(64, 3, padding = "same",
            name = "conv_final0", input_shape = (None, 64))
        
        self.conv_final = Conv1D(64, 3, 
            padding = "valid", name= "conv_final", 
            input_shape = (None, 64))
            
        self.gap1d = GlobalAveragePooling1D(name = "gap1d")
        
        self.dense2 = Dense(50, activation=activ, name="dense2")
        self.activ7 = Activation(activ, name="activ7")
        
        self.drop3 = Dropout(drop_rate, name="drop3")
        
        self.out = Dense(num_classes, kernel_initializer=init,
                         activation=activ_end, name="out1")
                         
    
    def branch(self, r, i, training):
        x1 = r[:,:,:,i,:]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.activ1(x1)
        x1 = self.drop1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.activ2(x2)
        
        x2 = self.conv3(x2)
        x2 = self.bn3(x2)
        
        b1 = self.conv_b1(x1)
        
        x2 = self.skip0([x2, b1])
        x2 = self.activ3(x2)
        x2 = self.drop2(x2)
        
        x3 = self.conv4(x2)
        x3 = self.bn4(x3)
        x3 = self.activ4(x3)
        
        x3 = self.conv5(x3)
        x3 = self.bn5(x3)
        
        b2 = self.conv_b2(x2)
        
        x3 = self.skip1([x3, b2])
        x3 = self.activ5(x3)
        x3 = self.gap(x3)
        
        return x3
        
        
    def call(self, inputs, training = True):
        
        out_list = []
        inputs = tf.cast(inputs, tf.float32)
        input1 = inputs[0]
        input2 = inputs[1]
        nb_slices = input1.shape[3]
        for i in range(nb_slices):
            out_list.append(self.branch(input1, i, training))
            
        x = self.conc(out_list)
        
        x = tf.expand_dims(x, axis = 0)

        x = self.conv_final0(x)
        x = self.conv_final(x)
        
        if nb_slices > 3:
            x = self.gap1d(x)
        else:
            x = tf.squeeze(x, axis =1)
        
        input2 = self.gap_scalars(input2)
        input2 = tf.expand_dims(input2[0], axis = 0)
        x = self.conc_scalars([x, input2])
        
        x = self.dense2(x)
        x = self.activ7(x)
        if training is True:
            x = self.drop3(x)
        output = self.out(x)
        
        return output
