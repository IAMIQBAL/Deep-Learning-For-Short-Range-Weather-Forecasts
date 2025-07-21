import sys
sys.path.append('models/')
sys.path.append('code/')
sys.path.append('utils/')

import base_model as bm
import derived_model as dm
import model_helpers as mh
import data_preproc as dpp
import evaluate as eval
import animation as anim
import plot as plt
from keras import losses
from keras import optimizers
import tensorflow as tf
import numpy as np
import yaml


#---------------------------------------#
#       Load Model Configurations       #
#---------------------------------------#

config_path = sys.argv[1]
print(config_path)

path = 'experiments/' + config_path
print(path)
with open(path + '/model_config.yaml', 'r') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

#---------------------------------------#
#           Initialize Model            #
#---------------------------------------#

n_layers = model_config['n_layers']
n_filters = model_config['n_filters']
kernel_size = [tuple(map(int, ks)) for ks in model_config['kernel_size']]
input_shape = tuple(model_config['input_shape'])

input_layer = tf.keras.Input(shape=input_shape)
if model_config['model'] == 'derived':
    x = dm.DerivedModel(n_layers, n_filters, kernel_size)(input_layer)
else:
    x = bm.BaseModel(n_layers, n_filters, kernel_size)(input_layer)

model = tf.keras.Model(inputs=input_layer, outputs=x)
print(model.summary(expand_nested=True))

#---------------------------------------#
#           Compile Model               #
#---------------------------------------#

model = mh.compile_model(model,
                        losses.mean_squared_error, 
                        tf.keras.optimizers.legacy.Adam(), 
                        ['accuracy', 'mae', 'mse'])

#---------------------------------------#
#           Train Model                 #
#---------------------------------------#

#---------------------------------------#
#           Predictiction               #
#---------------------------------------#

test_data = dpp.process_data(model_config['test_data_dir'])

model.load_weights(path + model_config['weights_dir'])

if model_config['variable'] == 'precipitation':
    cmap = plt.get_prec_cmap()
elif model_config['variable'] == 'temperature':
    cmap = plt.get_temp_cmap()
elif model_config['variable'] == 'Wind_U' or model_config['variable'] == 'Wind_V':
    cmap = 'BuGn'
act, pred = eval.predict(model, test_data[6:7, :, ...][0], 24, cmap, True)

act = act*255
pred = pred*255
print(model_config)
duration = 0.2
anim_dir = model_config['anim_dir']
anim.generateColoredGIF(act, cmap, anim_dir)
anim.generateColoredGIF(pred, cmap, anim_dir)