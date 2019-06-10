import os
from keras.models import load_model
from multi_slice import Multi_Resnet
from multi_slice_combined import Multi_Resnet_scalars

"""
Example for 2D model:
Load model with standard Keras function
Also works for 3D model
"""
model_name = "2D_CNN"
split = 1

model_path = os.path.join(model_name, "MR" + str(split), 
	"best_av_model.h5")
model = load_model(model_path)


"""
Example for multi-slice model:
Initialize model first, then load weights
"""
model_name = "Mutislice"
split = 1

model = Multi_Resnet(num_classes = 2)
model_path = os.path.join(model_name, "MS" + str(split), 
		"best_av_model")
model.load_weights(model_path)
		

"""
Example for multi-slice model including age and gender:
Initialize model first, then load weights
"""
model_name = "Multi_combined"
split = 1

model = Multi_Resnet_scalars(num_classes = 2)
model_path = os.path.join(model_name, "MS" + str(split), 
		"best_av_model")
model.load_weights(model_path)
	
