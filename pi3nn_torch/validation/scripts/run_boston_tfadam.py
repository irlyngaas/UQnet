"""
Runs the real pi3nn_torch pipeline (PI3NNTrainer, full Max_iter=5000,
early stopping, boundary optimization, all three networks) on boston
data, but with torch.optim.Adam swapped for the TF-exact-formula TFAdam,
to see how much of the gap against the TF reference run
(PICP_train/valid/test=0.9511/0.9348/0.8627, MPIW~1.2695,
RMSE_test=0.4228, R2_test=0.8142) is explained by the Adam-formula
difference identified in this session vs irreducible float noise.
"""
import sys
sys.path.insert(0, '/home/i0l/hydrogen-emulator-configurable')
sys.path.insert(0, '/home/i0l/UQnet/pi3nn_torch/validation/scripts')

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pi3nn_torch.data import load_boston
from pi3nn_torch.networks import UQNetMean, UQNetStd
import pi3nn_torch.trainer as trainer_mod
from tfadam_optim import TFAdam

# Monkeypatch: same signature as pi3nn_torch.trainer._make_optimizer,
# but routes 'Adam' to TFAdam instead of torch.optim.Adam.
def _make_tf_optimizer(name, params, lr):
    if name == 'Adam':
        return TFAdam(params, lr=lr, eps=1e-7)
    raise ValueError(name)


trainer_mod._make_optimizer = _make_tf_optimizer

seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

X, Y = load_boston()
Y = Y.reshape(-1, 1)

x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, Y, test_size=0.1, random_state=1, shuffle=True)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.1, random_state=1, shuffle=True)

scalar_x = StandardScaler()
scalar_y = StandardScaler()
x_train = scalar_x.fit_transform(x_train)
x_valid = scalar_x.fit_transform(x_valid)
x_test = scalar_x.transform(x_test)
y_train = scalar_y.fit_transform(y_train)
y_valid = scalar_y.fit_transform(y_valid)
y_test = scalar_y.transform(y_test)

num_inputs = x_train.shape[1]
num_outputs = 1

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

net_mean = UQNetMean(num_inputs, num_outputs, num_neurons=[50])
net_up = UQNetStd(num_inputs, num_outputs, num_neurons=[50])
net_down = UQNetStd(num_inputs, num_outputs, num_neurons=[50])

trainer = trainer_mod.PI3NNTrainer(
    configs, net_mean, net_up, net_down,
    x_train, y_train, x_valid, y_valid, x_test, y_test,
)
trainer.train()
trainer.boundary_optimization(verbose=1)
results = trainer.evaluate(final_evaluation=True, verbose=0)

print('-' * 40)
print('Results with TF-exact-formula Adam:')
for k, v in results.items():
    print(f'{k}: {v:.4f}')
print('-' * 40)
print('TF reference: picp_train=0.9511 picp_valid=0.9348 picp_test=0.8627 '
      'mpiw_train=1.2696 mpiw_valid=1.2695 mpiw_test=1.2694 rmse_test=0.4228 r2_test=0.8142')
