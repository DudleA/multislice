import os, glob, sys, io
import copy, csv
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks, losses, metrics
from tensorflow.keras.models import load_model

from generator import DataGenerator
from read import xReadFunction, yReadFunction, ReadPartition
from GetFileLists import GetFileLists
from network import Multi_CNN
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

# ---------------------------------------------------------------------------------
'DATASET AND PRE-PROCESSING'

'Dataset'
params["data"] = "dataset_name"

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
params_v["augmentation"] = [0, 0, 0, 0, 0]

# ---------------------------------------------------------------------------------
'LOSS & METRICS'

params["optimizer"] = Adam()
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
    params["augmentation"] = [0, 0, 0, 0]

# ---------------------------------------------------------------------------------
'PARTITION'
'Use existing partition: read, or randomly generated: None'
params["partition"] = 'read'
'For existing partition, choose index'
col = 0

if params["partition"] is None:
    partition = GetFileLists(params)
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
    balance = True)

validation_generator = DataGenerator(partition['validation'], params_v, 
    plotgenerator=0, batch_size=batch_size, shuffle=False, 
    balance = False)

# ----------------------------------------------------------------------------------
'MODEL'

'Load trained weights for 2D CNN'
old_model_path = " "
old_model = load_model(os.path.join(params["source"],
    old_model_path))

'Initialize multi-slice model'
model = Multi_CNN()
dummy_x = tf.zeros((1,2 * dx, 2 * dx, 6, 2), dtype = tf.dtypes.float64)
model._set_inputs(dummy_x)
#model.summary()

'Initialize the branch layers with the pre-trained weights'
for l in old_model.layers[:15]:
    w = l.get_weights()
    if len(w)==0:
        continue   
    for n in model.layers:
        if l.name in n.name:
            n.set_weights(w)
        
# ----------------------------------------------------------------------------------
'TRAINING'

loss = params["LossFunction"]
optimizer = params["optimizer"]
root = tf.train.Checkpoint(optimizer = optimizer, model = model)
    
'Create csv file to save metrics during training'
filename = 'metrics_training.csv'
csv_file = io.open(os.path.join(params["savefolder"],
    filename), 'w')
class CustomDialect(csv.excel):
    delimiter = ','
fieldnames = ['epoch', 'Training AUC', 'Training accuracy',
            'Training loss', 'Training sensitivity', 
            'Training specificity', 'Validation AUC', 
            'Validation accuracy', 'Validation loss', 
            'Validation sensitivity', 'Validation specificity']   
writer = csv.DictWriter(csv_file,
                            fieldnames=fieldnames,
                            dialect=CustomDialect)
    
writer.writeheader()
loss_val = []
    
for epoch in range(num_epochs):
    print("EPOCH ", epoch, "/", num_epochs)
    temp_loss_list = []
    label_list = []
    pred_list = []
    
    temp_loss_val = []
    label_val = []
    pred_val = []
    
    'Update weights on training set'
    for i in range(training_generator.__len__()):
        x_train,y_train=training_generator.__getitem__(i)
        with tf.GradientTape() as tape:
            logits = model(x_train, training = True)
            loss_values = loss(y_train, logits)
            
        'Follow training by printing the loss and prediction for each image'
        #print("Batch ", i, "/",training_generator.__len__(), 
        #    ", Loss: ", loss_values.numpy()[0]) 
        #print("Prediction: ", logits.numpy()[0,:], ", Label: ", y_train[0])

        temp_loss_list.append(loss_values.numpy().mean())
        label_list.append(y_train[0])
        pred_list.append(logits.numpy()[0,1])
        
        grads = tape.gradient(loss_values, model.trainable_variables[-4:])
        optimizer.apply_gradients(zip(grads, model.trainable_variables[-4:]))

    'Compute metrics on training set'
    loss_train = np.mean(np.asarray(temp_loss_list))
    auc_train = auc(label_list, pred_list)
    print("Training loss: ", loss_train, ", AUC: ", auc_train)
    
    acc, sens, spec = conf(label_list, pred_list)
    print("Training accuracy: ", acc, ", sensitivity: ", sens, ", specificity: ", spec)
        
    'Evaluate on validation set'
    for i in range(validation_generator.__len__()):
        x_train,y_train=validation_generator.__getitem__(i)
        logits = model(x_train, training = False)
        loss_values = loss(y_train, logits)
        
        #print("Batch ", i, "/", validation_generator.__len__(), 
        #    ", Loss: ", loss_values.numpy()[0])
        #print("Prediction: ", logits.numpy()[0,:], ", Label: ", y_train[0])

        temp_loss_val.append(loss_values.numpy().mean())
        label_val.append(y_train[0])
        pred_val.append(logits.numpy()[0,1])
        
    'Compute metrics on validation set'
    loss_val.append(np.mean(np.asarray(temp_loss_val)))
    auc_val = auc(label_val, pred_val)
    print("Validation loss: ", loss_val[-1], ", AUC: ", auc_val)
    
    acc_val, sens_val, spec_val = conf(label_val, pred_val)
    print("Validation accuracy: ", acc_val, ", sensitivity: ", sens_val,
        ", specificity: ", spec_val)
    
     'Save best model'
    if np.min(np.asarray(loss_val)) == loss_val[-1]:
        root.save(os.path.join(params["savefolder"], "best_model"))
        print("New best model")
    
    'Write to csv file'
    writer.writerow({'epoch': epoch, 
        'Training AUC': auc_train,  'Training accuracy': acc_train, 
        'Training loss': loss_train, 'Training sensitivity': sens_train, 
        'Training specificity': spec_train, 'Validation AUC': auc_val,
        'Validation accuracy': acc_val, 'Validation loss': loss_val[-1],
        'Validation sensitivity': sens_val, 
        'Validation specificity': spec_val})
    
    csv_file.flush()
    training_generator.on_epoch_end()
    validation_generator.on_epoch_end()
    
csv_file.close()

 'Save last model'
root.save(os.path.join(params["savefolder"], "last_model"))
