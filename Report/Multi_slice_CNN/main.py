import os, glob, sys, io
import copy, csv
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks, losses, metrics
from tensorflow.keras.models import load_model

from generator import DataGenerator
from read import xReadFunction, yReadFunction, scalarsReadFunction, ReadPartition
from GetFileLists_stratified import GetFileLists_stratified
from LossFunctions import dice_coef
from multi_slice import Multi_Resnet
from multi_slice_combined import Multi_Resnet_scalars
from metrics import auc, conf


params = {}
params["source"] = sys.argv[1]
params["savefolder"] = sys.argv[2]
params['xReadFunction'] = xReadFunction
params['yReadFunction'] = yReadFunction

# ---------------------------------------------------------------------------------
'GENERAL TRAINING PARAMETERS'

augmentation = True
batch_size = 1
num_epochs = 100

'Include age and gender'
params["scalars"] = False

'Train end to end or freeze branch layers'
params["endtoend"] = True

'Continue training'
model_load = False
start_epoch = None

# ---------------------------------------------------------------------------------
'DATASET AND PRE-PROCESSING'

'Dataset'
params["data"] = "dataset"

'Normalization: None, "clip", "scan" or "slice"'
params["norm"] = "slice"

'Spacing: "same" or "original"'
params["spacing"] = "original"

# ---------------------------------------------------------------------------------
'SHAPE'

'Crop shape: if dz is defined, the number of slices & branches is fixed'
dx = 180
#dz = 4
params["distance"] = (dx, dx, None)
params["shape"] = [1, 2 * dx, 2 * dx, 2]
input_shape = params["shape"][1:]

'Crop centering: "seg" or "scan"'
params["centered"] = "scan"

# ---------------------------------------------------------------------------------
'Copy parameters for validation generator'
params_v = copy.deepcopy(params)
params_v["augmentation"] = [0, 0, 0, 0]

# ---------------------------------------------------------------------------------
'LOSS & METRICS'

params["optimizer"] = Adam(lr = 0.00001)
params["LossFunction"] = losses.sparse_categorical_crossentropy

# ---------------------------------------------------------------------------------
'AUGMENTATION PARAMETERS'

if augmentation == True:
    print("Data augmentation")
    'Choose which augmentation methods are applied: '
    '[shift/rotation/flip/zoom, elastic deformations, gaussian noise, gamma correction]'
    params["augmentation"] = [1, 1, 1, 0]

    params["random_deform"] = dict()
    params["e_deform_g"] = dict()
    params["e_deform_p"] = dict()
    params["noise"] = dict()
    params["gamma"] = dict()

    'Shift, rotation, flip and zoom'
    params["random_deform"]['width_shift_range'] = 0.05
    params["random_deform"]['height_shift_range'] = 0.05
    params["random_deform"]['rotation_range_alpha'] = 20
    params["random_deform"]['horizontal_flip'] = False
    params["random_deform"]['vertical_flip'] = False
    params["random_deform"]['zoom_range'] = [0.9, 1.1]

    'Elastic deformations'
    params["e_deform_g"]["points"] = 3
    params["e_deform_g"]["sigma"] = 3

    'Gaussian noise'
    params["noise"]["mean"] = 0
    params["noise"]["std"] = 0.01

    'Gamma correction'
    params["gamma"]["range"] = [0, 2]
    params["gamma"]["mean"] = 1
    params["gamma"]["std"] = 0.2

else:
    params["augmentation"] = [0, 0, 0, 0, 0]

# ---------------------------------------------------------------------------------
'PARTITION'
'Use existing partition: read, or randomly generated: None'
params["partition"] = 'read'
'For existing partition, choose index'
col = 0

if params["partition"] is None:
    partition = GetFileLists_stratified(params)
    print("New partition")
    
elif params["partition"] == 'read':
    partition = ReadPartition(params["source"], params, col=col)
    if col is not None:
        print("Read partition n. ", col)
    else:
        print("Read random partition"
else:
    raise NameError("params['partition'] not defined")

# ---------------------------------------------------------------------------------
'DATA GENERATORS'

training_generator = DataGenerator(partition['train'], params, 
    plotgenerator=0, batch_size=batch_size, shuffle=True,
    balance = False)

validation_generator = DataGenerator(partition['validation'], params_v, 
    plotgenerator=0, batch_size=batch_size, shuffle=False, 
    balance = False)

# ----------------------------------------------------------------------------------
'MODEL'

old_model_path = " "

'Train merging and classification layers, without age and gender'
if (params["scalars"] is False) & (params["endtoend"] is False):
        nb_train_layers = 8
        
        'Initialize normal and EMA models'
        model = Multi_Resnet(num_classes = num_classes)
        av_model = Multi_Resnet(num_classes = num_classes)
        
        'Input: image and segmentation'
        if params["only"] == "both":
            dummy_x = tf.zeros((1,2 * dx, 2 * dx, 6, 2), 
                dtype = tf.dtypes.float64)
        
        'Input: image only'
        elif params["only"] == "im":
            dummy_x = tf.zeros((1,2 * dx, 2 * dx, 6, 1), 
                dtype = tf.dtypes.float64)
        else:
            raise ValueError("With or without segmentation ?")
        
        model._set_inputs(dummy_x)
        av_model._set_inputs(dummy_x)

        'Load pre-trained  weights from 2D model'
        if model_load is False:
            
            'Input: image and segmentation'
            if params["only"] == "both":
                old_model = load_model(os.path.join(params["source"],
                    "2D_CNN/best_av_model_col"+ str(col) + ".h5"), 
                    custom_objects = {"dice_coef": dice_coef})
            
            'Input: image only'
            elif params["only"] == "im":
                old_model = load_model(os.path.join(params["source"], 
                    "2D_CNN/best_av_model_col"+ str(col) + ".h5"), 
                    custom_objects = {"dice_coef": dice_coef})
            else:
                raise ValueError("With or without segmentation ?")
        
            'Initialize layers with pre-trained  weights from 2D model'
            for l in old_model.layers[:20]:
                #print(l.name)
                w = l.get_weights()
                if len(w)==0:
                    continue
                for n in model.layers:
                    if l.name in n.name:
                        #print(n.name)
                        n.set_weights(w)
                        
            'Initialize EMA model with same weights'
            for l in model.layers:
                w = l.get_weights()
                if len(w)==0:
                    continue
                for av_l in av_model.layers:
                    if l.name in av_l.name:
                        av_l.set_weights(w)
        else:
            'Continue training'
            model.load_weights(os.path.join(params["savefolder"], "last_model"))
            av_model.load_weights(os.path.join(params["savefolder"], "last_av_model"))
            
    'Train end to end, without age and gender'
    elif (params["scalars"] is False) & (params["endtoend"] is True):        
        
        'Initialize normal and EMA models'
        model = Multi_Resnet(num_classes = num_classes)
        av_model = Multi_Resnet(num_classes = num_classes)
        
        'Input: image and segmentation'
        if params["only"] == "both":
            dummy_x = tf.zeros((1,2 * dx, 2 * dx, 6, 2), 
                dtype = tf.dtypes.float64)
        
        'Input: image only'
        elif params["only"] == "im":
            dummy_x = tf.zeros((1,2 * dx, 2 * dx, 6, 1), 
                dtype = tf.dtypes.float64)
        else:
            raise ValueError("With or without segmentation ?")
        
        model._set_inputs(dummy_x)
        av_model._set_inputs(dummy_x)
        
        nb_train_layers = len(model.trainable_variables)
        
        'Load pre-trained weights from multi-slice model'
        if model_load is False:
            
            'Input: image and segmentation'
            if params["only"] == "both":
                model.load_weights(os.path.join(params["source"], 
                        "Multislice/Col"+str(col)+"/best_av_model"))
                av_model.load_weights(os.path.join(params["source"], 
                        "Multislice/Col"+str(col)+"/best_av_model"))
            
            'Input: image only'
            elif params["only"] == "im":
                model.load_weights(os.path.join(params["source"], 
                        "Images/Multislice_noseg/Col"+str(col)+"/best_av_model"))
                av_model.load_weights(os.path.join(params["source"], 
                        "Images/Multislice_noseg/Col"+str(col)+"/best_av_model"))
            else:
                raise ValueError("With or without segmentation ?")
        else:
            'Continue training'
            model.load_weights(os.path.join(params["savefolder"], "last_model"))
            av_model.load_weights(os.path.join(params["savefolder"], "last_av_model"))
    
    'Train classification layers with age and gender'
    else:
        nb_train_layers = 4
        
        'Initialize normal and EMA models'
        model = Multi_Resnet_scalars(num_classes = num_classes)
        av_model = Multi_Resnet_scalars(num_classes = num_classes)
        
        dummy_x = [tf.zeros((1,2 * dx, 2 * dx, 6, 2), 
            dtype = tf.dtypes.float64), 
            tf.zeros((1, 2 * dx, 2* dx, 6, 2), 
            dtype = tf.dtypes.float64)]
        model._set_inputs(dummy_x)
        av_model._set_inputs(dummy_x)        
        
        'Load pre-trained  weights from multi-slice model trained end to end'
        if model_load is False:
            old_model = Multi_Resnet(num_classes = num_classes)
            dummy_x = tf.zeros((1,2 * dx, 2 * dx, 6, 2), 
                dtype = tf.dtypes.float64)
            old_model._set_inputs(dummy_x)
            old_model.load_weights(os.path.join(params["source"], 
                "Multislice_endtoend/Col"+str(col)+"/best_av_model"))
            
            'Initialize layers with pre-trained  weights from multi-slice model trained end to end'
            for l in old_model.layers[:26]:
                w = l.get_weights()
                if len(w)==0:
                    continue
            
                for n in model.layers:
                    if l.name in n.name:
                        n.set_weights(w)
            
            'Initialize EMA model with same weights'
            for l in model.layers:
                w = l.get_weights()
                if len(w)==0:
                    continue
                for av_l in av_model.layers:
                    if l.name in av_l.name:
                        av_l.set_weights(w)
            
        "Continue training"
        else:
            model.load_weights(os.path.join(params["savefolder"], "last_model"))
            av_model.load_weights(os.path.join(params["savefolder"], "last_av_model"))
                    
    
# ----------------------------------------------------------------------------------
'TRAINING'

num_updates = float(start_epoch)
best = np.Inf

loss_object = params["LossFunction"]
optimizer = params["optimizer"]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
'Create csv file to save metrics during training'
filename = 'metrics_training.csv'
filename_ema = 'metrics_training_ema.csv'
    
if model_load is False:
    mode = 'w'
else :
    mode = 'a'

csv_file = io.open(os.path.join(params["savefolder"],
    filename), mode)
csv_file = io.open(os.path.join(params["savefolder"],
    filename_ema), mode)
    
class CustomDialect(csv.excel):
    delimiter = ','
    
fieldnames = ['epoch', 'Training AUC', 'Training accuracy',
            'Training loss', 'Training sensitivity', 
            'Training specificity', 'Validation AUC', 
            'Validation accuracy', 'Validation loss', 
            'Validation sensitivity', 'Validation specificity']   
            
fieldnames_ema = ['epoch', 'Validation AUC', 
            'Validation accuracy', 'Validation sensitivity', 
            'Validation specificity']
            
writer = csv.DictWriter(csv_file,
    fieldnames=fieldnames, dialect=CustomDialect)
                            
writer_ema = csv.DictWriter(csv_file_ema,
    fieldnames=fieldnames_ema, dialect=CustomDialect)

'Initialize lists'
loss_val = []
label_list = []
pred_list = []
    
label_val = []
pred_val = []


if model_load is False:
    writer.writeheader()
    writer_ema.writeheader()
    start_epoch = 0
else:
    df = pd.read_csv(os.path.join(params["savefolder"], 
        "metrics_training.csv"), header = 0, index_col = 0)
    loss_val.append(np.min(df.loc[:, "Validation loss"]))
    
    
"Start training"
for epoch in range(num_epochs):
    print("EPOCH ", epoch, "/", num_epochs)
    
    "Reset all metrics"
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    label_list = []
    pred_list = []
    label_val = []
    pred_val = []
    pred_val_ema = []
    
    "Iterate over all images of training set"
    for i in range(training_generator.__len__()):
        print("Batch ", i, "/",training_generator.__len__() - 1)
        x_train, y_train = training_generator.__getitem__(i)
        "Update model"
        with tf.GradientTape() as tape:
            predictions = model(x_train, training = True)
            loss = loss_object(y_train, predictions)
        gradients = tape.gradient(loss, 
            model.trainable_variables[-nb_train_layers:])
        optimizer.apply_gradients(zip(gradients, 
            model.trainable_variables[-nb_train_layers:]))
            
        "Collect labels and predictions"
        train_loss(loss)
        train_accuracy(y_train[0], predictions)
        label_list.append(y_train[0])
        pred_list.append(predictions.numpy()[0,1])
            
        "Update EMA model"
        current_decay = (1.0 + num_updates)/(10.0 + num_updates)
        for l in model.layers:
            new_w = l.get_weights()
            if len(new_w)==0:
                continue
            old_w = av_model.get_layer(l.name).get_weights()
            for i in range(len(old_w)):
                old_w[i] -= (1.0 - current_decay) * (old_w[i] - new_w[i])
            av_model.get_layer(l.name).set_weights(old_w)
        num_updates += 1.0
    
    "Compute metrics for training set"
    auc_train = auc(label_list, pred_list)        
    acc_train, sens_train, spec_train = conf(label_list, pred_list)
        
    "Iterate over all images of validation set"
    for i in range(validation_generator.__len__()):
        print("Batch ", i, "/",validation_generator.__len__() - 1)
        x_train, y_train = validation_generator.__getitem__(i)
        predictions = model(x_train, training = False)
        t_loss = loss_object(y_train, predictions)
    
        "Collect labels and predictions"
        test_loss(t_loss)
        test_accuracy(y_train[0], predictions)
        label_val.append(y_train[0])
        pred_val.append(predictions.numpy()[0,1])
        
        "Collect predictions for the EMA model"
        predictions_ema = av_model(x_train, training = False)
        pred_val_ema.append(predictions_ema.numpy()[0,1])
        
    "Compute metrics"
    auc_val = auc(label_val, pred_val)        
    acc_val, sens_val, spec_val = conf(label_val, pred_val)
    
    "Save latest model"
    model.save_weights(os.path.join(params["savefolder"], 
        "last_model"), save_format = 'tf')
    
    "Update best model if new minimum for validation loss"
    loss_val.append(test_loss.result().numpy())
    if np.min(np.asarray(loss_val)) == loss_val[-1]:
        model.save_weights(os.path.join(params["savefolder"], 
            "best_model"), save_format = 'tf')
        print("New best model")
    
        
    print(template.format(epoch, train_loss.result(), 
        train_accuracy.result(), auc_train, sens_train, spec_train))
    print(template_val.format(test_loss.result(), 
        test_accuracy.result(), auc_val, sens_val, spec_val))
    
    "Write to csv file"
    writer.writerow({'epoch': epoch+start_epoch, 
        'Training AUC': auc_train, 
        'Training accuracy': train_accuracy.result().numpy(), 
        'Training loss': train_loss.result().numpy(), 
        'Training sensitivity': sens_train, 
        'Training specificity': spec_train,
        'Validation AUC': auc_val,
        'Validation accuracy': test_accuracy.result().numpy(),
        'Validation loss': test_loss.result().numpy(),
        'Validation sensitivity': sens_val,
        'Validation specificity': spec_val})
    csv_file.flush()
        
    "Compute metrics for EMA model"
    auc_val_ema = auc(label_val, pred_val_ema)        
    acc_val_ema, sens_val_ema, spec_val_ema = conf(label_val, 
        pred_val_ema)
        
    "Save latest EMA model"
    av_model.save_weights(os.path.join(params["savefolder"], 
        "last_av_model"), save_format = 'tf')
    
    "Update best EMA model if new minimum for validation loss"
    loss_val.append(test_loss.result().numpy())
    if np.min(np.asarray(loss_val)) == loss_val[-1]:
        av_model.save_weights(os.path.join(params["savefolder"], 
            "best_av_model"), save_format = 'tf')
            
    print("EMA: ", template_val_ema.format(acc_val_ema, auc_val_ema, 
        sens_val_ema, spec_val_ema))
        
    "Write to csv file"
    writer_ema.writerow({'epoch': epoch+start_epoch, 
        'Validation AUC': auc_val_ema,
        'Validation accuracy': acc_val_ema,
        'Validation sensitivity': sens_val_ema,
        'Validation specificity': spec_val_ema})
    csv_file_ema.flush()
        
    training_generator.on_epoch_end()
    validation_generator.on_epoch_end()
    
csv_file.close()
csv_file_ema.close()
