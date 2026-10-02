"""
The cleanest possible apples-to-apples test: load TF's exact initial
weights for net_mean, net_up, AND net_down (saved earlier in
tf_init_state.npz) into the PyTorch port, use TF's exact Adam formula
(TFAdam), use the TF-saved preprocessed data, then run the real
pi3nn_torch.trainer.PI3NNTrainer pipeline with early stopping enabled
exactly as main_PI3NN.py's manual mode does (Max_iter=5000,
early_stop_start_iter=100, wait_patience=300), and compare the final
PICP/MPIW/RMSE/R2 directly against the real TF run's numbers.

If this lands very close to the TF reference, it confirms the port's
algorithm is correct and essentially all of the original ~12% gap was
(a) different random initialization and (b) the Adam-formula difference,
not a structural port bug.
"""
import sys
sys.path.insert(0, '/home/i0l/hydrogen-emulator-configurable')
sys.path.insert(0, '/home/i0l/UQnet/pi3nn_torch/validation/scripts')

import numpy as np
import torch

from pi3nn_torch.networks import UQNetMean, UQNetStd
import pi3nn_torch.trainer as trainer_mod
from tfadam_optim import TFAdam

def _make_tf_optimizer(name, params, lr):
    if name == 'Adam':
        return TFAdam(params, lr=lr, eps=1e-7)
    raise ValueError(name)

trainer_mod._make_optimizer = _make_tf_optimizer

DATA = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_init_state.npz')

x_train, y_train = DATA['xTrain'], DATA['yTrain']
x_valid, y_valid = DATA['xValid'], DATA['yValid']
x_test, y_test = DATA['xTest'], DATA['yTest']


def load_mean(net, prefix):
    with torch.no_grad():
        net.input_layer.weight.copy_(torch.as_tensor(DATA[f'{prefix}_input_kernel'].T))
        net.input_layer.bias.copy_(torch.as_tensor(DATA[f'{prefix}_input_bias']))
        for i, fc in enumerate(net.fcs):
            fc.weight.copy_(torch.as_tensor(DATA[f'{prefix}_fc{i}_kernel'].T))
            fc.bias.copy_(torch.as_tensor(DATA[f'{prefix}_fc{i}_bias']))
        net.output_layer.weight.copy_(torch.as_tensor(DATA[f'{prefix}_output_kernel'].T))
        net.output_layer.bias.copy_(torch.as_tensor(DATA[f'{prefix}_output_bias']))


def load_std(net, prefix):
    load_mean(net, prefix)
    with torch.no_grad():
        net.custom_bias.copy_(torch.as_tensor(DATA[f'{prefix}_custom_bias'][0]))


net_mean = UQNetMean(13, 1, num_neurons=[50])
load_mean(net_mean, 'mean')
net_up = UQNetStd(13, 1, num_neurons=[50])
load_std(net_up, 'up')
net_down = UQNetStd(13, 1, num_neurons=[50])
load_std(net_down, 'down')

configs = {
    'quantile': 0.95,
    'verbose': 1,
    'Max_iter': 5000,
    'lr': [0.02, 0.02, 0.02],
    'optimizers': ['Adam', 'Adam', 'Adam'],
    'exponential_decay': True,
    'decay_steps': 3000,
    'decay_rate': 0.9,
    'early_stop': True,
    'early_stop_start_iter': 100,
    'wait_patience': 300,
    'restore_best_weights': True,
}

trainer = trainer_mod.PI3NNTrainer(
    configs, net_mean, net_up, net_down,
    x_train, y_train, x_valid, y_valid, x_test, y_test,
)
trainer.train()
trainer.boundary_optimization(verbose=1)
results = trainer.evaluate(final_evaluation=True, verbose=0)

print('-' * 40)
print('Results with matched TF init weights + TF-exact Adam:')
for k, v in results.items():
    print(f'{k}: {v:.4f}')
print('-' * 40)
print('TF reference: picp_train=0.9511 picp_valid=0.9348 picp_test=0.8627 '
      'mpiw_train=1.2696 mpiw_valid=1.2695 mpiw_test=1.2694 rmse_test=0.4228 r2_test=0.8142')
