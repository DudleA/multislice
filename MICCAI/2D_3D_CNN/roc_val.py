from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import csv


class roc_val(Callback):
    """
    Save ROC curve, metrics (accuracy, loss, AUC, sensitivity, 
    specificity) and predictions after each epoch for the validation set 
    """
    def __init__(self,generator, filepath, filename, label = None, 
        append = False, start_epoch = 0, separator=','):

        self.generator = generator
        
        self.sep = separator
        self.filepath = filepath
        self.filename = filename
        self.append = append
        self.start_epoch = start_epoch
        self.label = label
        
        self.writer = None
        self.dict_writer = None
        
        
    def on_train_begin(self, logs={}):
        """
        Create 3 csv files
        """
        if self.append is False:
            mode = 'w'
        else :
            mode = 'a'
    
        self.csv_file_roc = io.open(os.path.join(self.filepath,
            'roc'+self.filename),
                            mode)
        self.csv_file_auc = io.open(os.path.join(self.filepath,
            'auc'+self.filename),
                            mode)
        self.csv_file_pred = io.open(os.path.join(self.filepath,
            'pred'+self.filename),
                            mode)
                            
        if (not self.writer) or (not self.dict_writer):
            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            if self.label=="pheno":
                fieldnames = ['epoch', 'AUC', 'acc']
            else:
                fieldnames = ['epoch', 'AUC', 'acc','loss',
                    'sensitivity', 'specificity']   
            self.writer = csv.writer(self.csv_file_roc,
                                         dialect=CustomDialect)
            self.dict_writer = csv.DictWriter(self.csv_file_auc,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            self.pred_writer = csv.writer(self.csv_file_pred,
                                         dialect=CustomDialect)
        
        if self.append is False:
            self.dict_writer.writeheader()
            self.pred_writer.writerow(np.append(['full_list']
                ,self.generator.list_IDs))
        
    def on_train_end(self, logs={}):
        """
        Close csv files
        """
        
        self.csv_file_roc.close()
        self.csv_file_auc.close()
        self.csv_file_pred.close()
        self.writer = None
        self.dict_writer = None
        self.pred_writer = None

        return
    
    def on_epoch_end(self, epoch, logs={}):
        """
        Save results after each epoch
        """
        IDs = np.empty(0)
        
        'Collect labels and predidictions'
        for i in range(self.generator.__len__()):
            self.model._function_kwargs = {}
            x, y = self.generator.__getitem__(i)
            IDs = np.append(IDs, 
                np.asarray(self.generator.list_IDs_temp))
            y_pred = self.model.predict(x, steps = 1)
            if i==0:
                Y = y
                Y_pred = y_pred
                continue
            else:
                Y = np.append(Y,y,axis = 0)
                Y_pred = np.append(Y_pred,y_pred, axis =0)

        Y= Y.astype(int)
        strepoch = str(epoch+self.start_epoch)
        Y_pred = Y_pred[:,1]
        
        'Compute and print AUC'
        auc = roc_auc_score(Y, Y_pred)
        print("Validation AUC: ", auc)
        
        'Compute accuracy, loss, sensitivity and specificity'
        conf = confusion_matrix(Y, np.round(Y_pred))
        acc = np.trace(conf)/float(np.sum(conf))
        loss = -np.average(np.multiply(Y, np.log(Y_pred))+
            np.multiply(1-Y,np.log(1-Y_pred)))
        sensitivity = conf[1,1]/(conf[1,1]+conf[1,0])
        specificity = conf[0,0]/(conf[0,0]+conf[0,1])
        
        'Compute ROC curve'
        (fpr,tpr,thresholds) = roc_curve(Y, Y_pred)
        fpr = np.append(['fpr'+strepoch], fpr)
        tpr = np.append(['tpr'+strepoch], tpr)
        thresholds = np.append(['thresh'+strepoch], thresholds)
    
        'Write metrics to file'
        self.dict_writer.writerow({'epoch': epoch+self.start_epoch, 
            'AUC': auc, 'acc': acc, 'loss': loss,
            'sensitivity': sensitivity, 'specificity': specificity})
        self.csv_file_auc.flush()
    
        'Write ROC curve to file'
        self.writer.writerows([fpr, tpr, thresholds])
        self.csv_file_roc.flush()
        
        'Write predictions to file'
        IDs = np.append(['pred_list'+strepoch], IDs)
        Y = np.append(['Label'+strepoch], Y)
        Y_pred = np.append(['Pred'+strepoch], Y_pred)
        self.pred_writer.writerows([IDs, Y, Y_pred])
        self.csv_file_pred.flush()
