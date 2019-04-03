import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, Conv1D, 
    Input, Activation, Dropout, Concatenate, MaxPooling2D, 
    GlobalAveragePooling1D, GlobalMaxPooling2D, BatchNormalization)
from keras.activations import relu
import numpy as np


class Multi_CNN(tf.keras.Model):
    """
    Multi-slice CNN with scalable number of branches
    """
    
    def __init__(self, num_classes=2):
        """
        Initialize all layers
        """
        super(Multi_CNN, self).__init__()
        
        activ = relu
        activ_end = 'softmax'
        init = 'normal'
        dropout_rate = 0.3
        nfeat = 32
        dil_rate = 2
        
        self.conv1 = Conv2D(nfeat, (3, 3), kernel_initializer=init,
            padding='same', dilation_rate=dil_rate, name="conv1", 
            input_shape = (360,360,2))

        self.bn1 = BatchNormalization(name="bn1")
        self.activ1 = Activation(activ, name="activ1")

        self.conv2 = Conv2D(nfeat, (3, 3), kernel_initializer=init,
            padding='same', dilation_rate=dil_rate, name="conv2")

        self.bn2 = BatchNormalization(name="bn2")
        self.activ2 = Activation(activ, name="activ2")

        self.skip0 = Concatenate(axis=-1, name = "skip0")
        self.pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")
        self.drop1 = Dropout(dropout_rate, name="drop1")

        self.conv5 = Conv2D(nfeat * 2, (3, 3), kernel_initializer=init,
            padding='same', dilation_rate=dil_rate, name="conv5")
        self.bn5 = BatchNormalization(name="bn5")
        self.activ5 = Activation(activ, name="activ5")

        self.conv6 = Conv2D(nfeat * 2, (3, 3), kernel_initializer=init,
            padding='same', dilation_rate=dil_rate, name="conv6")
        self.bn6 = BatchNormalization(name="bn6")
        self.activ6 = Activation(activ, name="activ6")

        self.skip1 = Concatenate(axis=-1, name = "skip1")
        self.gmp = GlobalMaxPooling2D(name="gmp")
        self.drop2 = Dropout(dropout_rate, name="drop2")   
        
        self.conc = Concatenate(axis = 0, name = "merge")        
        self.conv_final = Conv1D(128, 3, kernel_initializer = init, 
            padding = "valid", name= "conv_final", 
            input_shape = (None, 128))
        self.gap1d = GlobalAveragePooling1D(name = "gap1d")
        
        self.dense = Dense(50, activation=activ, name="dense2")
        self.activ7 = Activation(activ, name="activ7")
        self.drop3 = Dropout(dropout_rate, name="drop3")
        self.out = Dense(num_classes, kernel_initializer=init,
                         activation=activ_end, name="out1")
                         
    def branch(self, r, i, training):
        """
        Define branch structure
        """
    
        x1 = r[:,:,:,i,:]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.activ1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.activ2(x2)
        
        x2 = self.skip0([x1,x2])
        x2 = self.pool1(x2)
        
        if training is True:
            y1 = self.drop1(x2)
            y1 = self.conv5(y1)
        else:
            y1 = self.conv5(x2)
            
        y1 = self.bn5(y1)
        y1 = self.activ5(y1)
        
        y1 = self.conv6(y1)
        y1 = self.bn6(y1)
        y1 = self.activ6(y1)
        
        y1 = self.skip1([x2, y1])
        y1 = self.gmp(y1)
        
        if training is True:
            y1 = self.drop2(y1)
        return y1
        
    
    def call(self, inputs, training = True):
        """
        Call function of model, applied to each input
        """
        out_list = []
        inputs = tf.cast(inputs, tf.float32)
        
        'Apply branch function to each slice'
        for i in range(inputs.shape[3]):
            out_list.append(self.branch(inputs, i, training))

        'Merging layers'
        x = self.conc(out_list)
        x = tf.expand_dims(x, axis = 0)
        x = self.conv_final(x)
        x = self.gap1d(x)
        
        'Classification layers'
        x = self.dense(x)
        x = self.activ7(x)
        if training is True:
            x = self.drop3(x)
        output = self.out(x)
        
        return output
