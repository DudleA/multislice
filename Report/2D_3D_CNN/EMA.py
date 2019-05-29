"""
Callback function to compute an exponential moving average of the 
model's weights while training, and store the weights of the last and 
best models (based on the validation loss).
In addition, the values of different metrics for the EMA model are
also stored : AUC, accuracy, sensitivity and specificity
"""
import numpy as np
import scipy.sparse as sp

from keras import backend as K
from keras.callbacks import Callback
from keras.models import load_model
import io, csv
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
#import tensorflow as tf

import sys, os
import warnings

from mini_resnet import mini_resnet
from mini_resnet3d import mini_resnet3d
from LossFunctions import binarized_dice_loss, dice_coef_loss, dice_coef, f1

class ExponentialMovingAverage(Callback):
    def __init__(self, input_shape, generator, filepath, filename, 
            scalars = False, num_classes = 2, append = False,  
            start_epoch = 0, separator=',', decay=0.999):
        
        self.decay = decay
                
        self.generator = generator
        self.sep = separator
        self.filepath = filepath
        self.filename = filename
        self.append = append
        self.start_epoch = start_epoch
        self.num_updates = float(start_epoch)
        self.num_classes = num_classes
        
        self.writer = None
        self.dict_writer = None
        self.best = np.Inf
        
        if self.append is False:
            if len(input_shape) == 3:
                self.av_model = mini_resnet(input_shape, 
                    scalars = scalars, num_classes = num_classes)
            else:
                self.av_model = mini_resnet3d(input_shape, 
                    scalars = scalars, num_classes = num_classes)
        else:
            self.av_model = load_model(os.path.join(filepath, 
                "last_av_model.h5"), custom_objects = 
                {'dice_coef': dice_coef})

        super(ExponentialMovingAverage, self).__init__()
        

    def on_train_begin(self, logs={}):
        
        if self.append is False:
            mode = 'w'
            for l in self.model.layers:
                w = l.get_weights()
                if len(w)==0:
                    continue
                for av_l in self.av_model.layers:
                    if l.name in av_l.name:
                        av_l.set_weights(w)
        else :
            mode = 'a'
            
        self.csv_file_roc = io.open(os.path.join(self.filepath,
            'roc'+self.filename), mode)
        self.csv_file_auc = io.open(os.path.join(self.filepath,
            'auc'+self.filename), mode)
        self.csv_file_pred = io.open(os.path.join(self.filepath,
            'pred'+self.filename), mode)
                            
        if (not self.writer) or (not self.dict_writer):
            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            if self.num_classes > 2:
                fieldnames = ['epoch', 'AUC', 'acc']
            else:
                fieldnames = ['epoch', 'AUC', 'acc', 'sensitivity', 
                    'specificity']   
              
            self.writer = csv.writer(self.csv_file_roc,
                                         dialect=CustomDialect)
            self.dict_writer = csv.DictWriter(self.csv_file_auc,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            self.pred_writer = csv.writer(self.csv_file_pred,
                                         dialect=CustomDialect)
        
        if self.append is False:
            self.dict_writer.writeheader()
            self.pred_writer.writerow(np.append(['full_list'],
                self.generator.list_IDs))


    def on_batch_end(self, batch, logs={}):
        current_decay = (1.0 + self.num_updates)/(10.0 + self.num_updates)
        for l in self.model.layers:
            new_w = l.get_weights()
            if len(new_w)==0:
                continue
            old_w = self.av_model.get_layer(l.name).get_weights()
            for i in range(len(old_w)):
                old_w[i] -= (1.0 - current_decay) * (old_w[i] - new_w[i])
            self.av_model.get_layer(l.name).set_weights(old_w)
        self.num_updates += 1.0
        
    def on_epoch_end(self, epoch, logs={}):
        
        IDs = np.empty(0)
        for i in range(self.generator.__len__()):
            x, y = self.generator.__getitem__(i)
            IDs = np.append(IDs, 
                np.asarray(self.generator.list_IDs_temp))
            
            y_pred = self.av_model.predict(x, steps = 1)

            if i==0:
                Y = y
                Y_pred = y_pred
                continue
            else:
                Y = np.append(Y,y,axis = 0)
                Y_pred = np.append(Y_pred,y_pred, axis =0)

        Y= Y.astype(int)
        strepoch = str(epoch+self.start_epoch)
        
        if self.num_classes > 2:
            auc = roc_auc_score(Y, Y_pred)            
            conf = confusion_matrix(np.argmax(Y, axis=1), np.argmax(Y_pred,axis=1))
            acc = np.trace(conf)/float(np.sum(conf))
            self.dict_writer.writerow({'epoch' : epoch+self.start_epoch,
                'AUC' : auc, 'acc': acc})
        else:
            Y_pred = Y_pred[:,1]
            auc = roc_auc_score(Y, Y_pred)
            conf = confusion_matrix(Y, np.round(Y_pred))
            acc = np.trace(conf)/float(np.sum(conf))
            sensitivity = conf[1,1]/(conf[1,1]+conf[1,0])
            specificity = conf[0,0]/(conf[0,0]+conf[0,1])
            (fpr,tpr,thresholds) = roc_curve(Y, Y_pred)
            fpr = np.append(['fpr'+strepoch], fpr)
            tpr = np.append(['tpr'+strepoch], tpr)
            thresholds = np.append(['thresh'+strepoch], thresholds)
            self.writer.writerows([fpr, tpr, thresholds])
            self.csv_file_roc.flush()
            
            self.dict_writer.writerow({'epoch': epoch+self.start_epoch, 
                'AUC': auc, 
                'acc': acc,
                'sensitivity': sensitivity, 
                'specificity': specificity})
        print("EMA validation AUC: ", auc)
        self.csv_file_auc.flush()

        IDs = np.append(['pred_list'+strepoch], IDs)
        Y = np.append(['Label'+strepoch], Y)
        Y_pred = np.append(['Pred'+strepoch], Y_pred)

        self.pred_writer.writerows([IDs, Y, Y_pred])
        self.csv_file_pred.flush()

        current = logs.get('val_loss')
        if np.less(current, self.best):
            self.best = current
            self.av_model.save(os.path.join(self.filepath, 
                "best_av_model.h5"), overwrite = True)


        if self.append is False:
            self.av_model.save(os.path.join(self.filepath, 
                "last_av_model.h5"))
        else:
            self.av_model.save(os.path.join(self.filepath, 
                "last_av_model2.h5"))
