import os, glob, sys
import copy

from keras.optimizers import Adam
from keras import optimizers, callbacks, metrics, losses
from keras.models import load_model

from generator import DataGenerator
from read import xReadFunction, yReadFunction, ReadPartition
from GetFileLists import GetFileLists
from network_2Dcnn import network_2Dcnn
from network_3Dcnn import network_3Dcnn
from roc_val import roc_val


params = {}
params["source"] = sys.argv[1]
params["savefolder"] = sys.argv[2]
params['xReadFunction'] = xReadFunction
params['yReadFunction'] = yReadFunction

# ---------------------------------------------------------------------------------
'GENERAL TRAINING PARAMETERS'
which_model = network_2dcnn
batch_size = 30
num_epochs = 100
augmentation = True

'Continue training'
model_load = False
start_epoch = None

# ---------------------------------------------------------------------------------
'DATASET AND PRE-PROCESSING'
'Dataset'
data = 'dataset_name'

'Normalization: None, "clip", "scan" or "slice"'
params["norm"] = "slice"

'Spacing: "same" or "original"'
params["spacing"] = "original"

# ---------------------------------------------------------------------------------
'SHAPE'

'Crop shape: if dz is defined, the number of slices & branches is fixed'
dx = 180
params["distance"] = (dx, dx)
params["shape"] = [1, 2 * dx, 2 * dx, 2]
input_shape = params["shape"][1:]

'Crop centering: "seg" or "scan"'
params["centered"] = "scan"

# ---------------------------------------------------------------------------------
'Copy parameters for validation generator'
params_v = copy.deepcopy(params)
params_v["augmentation"] = [0, 0, 0]

# ---------------------------------------------------------------------------------
'LOSS & METRICS'

params["optimizer"] = Adam()
params["LossFunction"] = losses.sparse_categorical_crossentropy
params["accuracy"] = [metrics.sparse_categorical_accuracy]

# ---------------------------------------------------------------------------------
'AUGMENTATION PARAMETERS'

if augmentation == True:
    print("Data augmentation")
    'Choose which augmentation methods are applied: '
    '[shift/rotation/flip/zoom, elastic deformations, gaussian noise, gamma correction]'
    params["augmentation"] = [1,, 1, 1, 0]

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

    params["e_deform_p"]["alpha"] = 10
    params["e_deform_p"]["sigma"] = 3

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
    partition = GetFileLists(params)
    print("New partition")
    
elif params["partition"] == 'read':
    partition = ReadPartition(params["source"], col=col,
        label=params["label"], data=params["data"],
        spacing=params["spacing"])
    if col is not None:
        print("Read partition n. ", col)
    else:
        print("Read random partition")
else:
    raise NameError("params['partition'] not defined")

# ---------------------------------------------------------------------------------
'DATA GENERATORS'

training_generator = DataGenerator(partition['train'], params, 
    plotgenerator=0, batch_size=batch_size, shuffle=True)

validation_generator = DataGenerator(partition['validation'], params_v, 
    plotgenerator=0, batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------------------------------
'MODEL'

'Initialize new model'
if model_load is False:
    model = which_model(input_shape)
'Continue training an existing model'
else:
    model = load_model(os.path.join(params["savefolder"], 
        "last_model.hdf5"))

model.compile(optimizer=params["optimizer"], 
    loss=params["LossFunction"],
    metrics=params['accuracy'])

# ----------------------------------------------------------------------------------
'CALLBACKS'

'Define new generator for callback'
callback_generator = DataGenerator(partition['validation'], params_v, 
    plotgenerator=0, batch_size=batch_size, shuffle=False)

if model_load is False:
    
    'Save accuracy & loss'
    csv_logger = callbacks.CSVLogger(os.path.join(params["savefolder"], 
        str(datetime.date.today()) + '-' + 
        str(datetime.datetime.now())[11:16] + 'training.csv'))

    'Custom callback'
    roc_callback = roc_val(callback_generator, params["savefolder"], 
        str(datetime.date.today()) + '-' + 
        str(datetime.datetime.now())[11:16] + 'roc_callback.csv', 
        label=params["label"])
    
    'Save last model'
    checkpoint = callbacks.ModelCheckpoint(os.path.join(
        params["savefolder"], "last_model.hdf5"))
    
    'Save best model'
    checkpoint_best = callbacks.ModelCheckpoint(os.path.join(
        params["savefolder"], "best_model.hdf5"), save_best_only=True)

else:
    'Append to existing files'
    
    csvfile = glob.glob(os.path.join(params["savefolder"],
        "*training.csv"))
    csv_logger = callbacks.CSVLogger(csvfile[0], append=True)

    roc_valfiles = glob.glob(os.path.join(params["savefolder"], 
        "*roc_callback.csv"))
    roc_valfile0 = roc_valfiles[0]
    roc_callback = roc_val(test_generator, params["savefolder"], 
        roc_valfile0[-32:], append=True, start_epoch=start_epoch)
        
    checkpoint = callbacks.ModelCheckpoint(os.path.join(
        params["savefolder"], "last_model.hdf5"))
    checkpoint_best = callbacks.ModelCheckpoint(os.path.join(
        params["savefolder"], "best_model.hdf5"), save_best_only=True)

# ----------------------------------------------------------------------------------
'TRAINING'

model.fit_generator(generator=training_generator,
                    verbose=1, epochs=num_epochs,
                    validation_data=validation_generator,
                    callbacks=[roc_callback, csv_logger, 
                    checkpoint, checkpoint_best])

