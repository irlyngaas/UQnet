"""
Run inside the TF venv. Reproduces main_PI3NN.py's exact boston data
split/scaling and builds the three TF networks with the same seed (10),
then saves:
  - the preprocessed train/valid/test arrays (so the torch side uses
    byte-identical data, removing sklearn-version drift as a variable)
  - each network's freshly-initialized weights (before any training)
so the torch port can be loaded with the SAME starting point and compared
on a fixed-iteration, no-early-stopping trajectory.
"""
import os
import random
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, '/home/i0l/UQnet')
from pi3nn.Networks.networks import UQ_Net_mean_TF2, UQ_Net_std_TF2
from pi3nn.DataLoaders.data_loaders import CL_dataLoader

seed = 10
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dataLoader = CL_dataLoader(original_data_path='/home/i0l/UQnet/datasets/UCI_datasets/')
X, Y = dataLoader.load_single_dataset('boston')
if len(Y.shape) == 1:
    Y = Y.reshape(-1, 1)

xTrainValid, xTest, yTrainValid, yTest = train_test_split(X, Y, test_size=0.1, random_state=1, shuffle=True)
xTrain, xValid, yTrain, yValid = train_test_split(xTrainValid, yTrainValid, test_size=0.1, random_state=1, shuffle=True)

scalar_x = StandardScaler()
scalar_y = StandardScaler()
xTrain = scalar_x.fit_transform(xTrain)
xValid = scalar_x.fit_transform(xValid)
xTest = scalar_x.transform(xTest)
yTrain = scalar_y.fit_transform(yTrain)
yValid = scalar_y.fit_transform(yValid)
yTest = scalar_y.transform(yTest)

num_inputs = xTrain.shape[1]
num_outputs = 1

configs = {
    'num_neurons_mean': [50],
    'num_neurons_up': [50],
    'num_neurons_down': [50],
}

net_mean = UQ_Net_mean_TF2(configs, num_inputs, num_outputs)
net_up = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='up')
net_down = UQ_Net_std_TF2(configs, num_inputs, num_outputs, net='down')

# Force weight creation (Keras builds lazily on first call).
_ = net_mean(xTrain[:1].astype(np.float32))
_ = net_up(xTrain[:1].astype(np.float32))
_ = net_down(xTrain[:1].astype(np.float32))


def dump_weights(net, prefix, has_bias_var=False):
    out = {}
    out[f'{prefix}_input_kernel'] = net.inputLayer.kernel.numpy()
    out[f'{prefix}_input_bias'] = net.inputLayer.bias.numpy()
    for i, fc in enumerate(net.fcs):
        out[f'{prefix}_fc{i}_kernel'] = fc.kernel.numpy()
        out[f'{prefix}_fc{i}_bias'] = fc.bias.numpy()
    out[f'{prefix}_output_kernel'] = net.outputLayer.kernel.numpy()
    out[f'{prefix}_output_bias'] = net.outputLayer.bias.numpy()
    if has_bias_var:
        out[f'{prefix}_custom_bias'] = net.custom_bias.numpy()
    return out


weights = {}
weights.update(dump_weights(net_mean, 'mean'))
weights.update(dump_weights(net_up, 'up', has_bias_var=True))
weights.update(dump_weights(net_down, 'down', has_bias_var=True))

np.savez(
    '/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_init_state.npz',
    xTrain=xTrain, yTrain=yTrain, xValid=xValid, yValid=yValid, xTest=xTest, yTest=yTest,
    **weights,
)
print('Saved data + initial weights.')
for k, v in weights.items():
    print(k, v.shape)
