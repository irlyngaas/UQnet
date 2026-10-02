"""
float64 variant of torch_matched_run.py, mirroring tf_matched_run_f64.py.
"""
import sys
sys.path.insert(0, '/home/i0l/hydrogen-emulator-configurable')

import numpy as np
import torch
import torch.nn as nn

from pi3nn_torch.networks import UQNetMean

DATA = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_init_state.npz')

xTrain = torch.as_tensor(DATA['xTrain'], dtype=torch.float64)
yTrain = torch.as_tensor(DATA['yTrain'], dtype=torch.float64).reshape(-1, 1)
xValid = torch.as_tensor(DATA['xValid'], dtype=torch.float64)
yValid = torch.as_tensor(DATA['yValid'], dtype=torch.float64).reshape(-1, 1)

net_mean = UQNetMean(13, 1, num_neurons=[50]).double()

with torch.no_grad():
    net_mean.input_layer.weight.copy_(torch.as_tensor(DATA['mean_input_kernel'].T, dtype=torch.float64))
    net_mean.input_layer.bias.copy_(torch.as_tensor(DATA['mean_input_bias'], dtype=torch.float64))
    net_mean.fcs[0].weight.copy_(torch.as_tensor(DATA['mean_fc0_kernel'].T, dtype=torch.float64))
    net_mean.fcs[0].bias.copy_(torch.as_tensor(DATA['mean_fc0_bias'], dtype=torch.float64))
    net_mean.output_layer.weight.copy_(torch.as_tensor(DATA['mean_output_kernel'].T, dtype=torch.float64))
    net_mean.output_layer.bias.copy_(torch.as_tensor(DATA['mean_output_bias'], dtype=torch.float64))

print('mean(x0):', net_mean(xTrain[:1]).item())

LR = 0.02
MAX_ITER = 1200
DECAY_STEPS = 3000
DECAY_RATE = 0.9

optimizer = torch.optim.Adam(net_mean.parameters(), lr=LR, eps=1e-7)
gamma = DECAY_RATE ** (1.0 / DECAY_STEPS)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
loss_fn = nn.MSELoss()

train_losses = []
valid_losses = []
for it in range(MAX_ITER):
    net_mean.train()
    optimizer.zero_grad()
    train_pred = net_mean(xTrain)
    train_loss = loss_fn(train_pred, yTrain) + net_mean.regularizer_loss()
    train_loss.backward()
    optimizer.step()
    scheduler.step()

    net_mean.eval()
    with torch.no_grad():
        valid_pred = net_mean(xValid)
        valid_loss = loss_fn(valid_pred, yValid).item()

    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss)
    if it % 100 == 0:
        print(f'[mean-f64] iter {it}, train_loss {train_loss.item():.10e}, valid_loss {valid_loss:.10e}')

np.savez(
    '/home/i0l/UQnet/pi3nn_torch/validation/artifacts/torch_matched_mean_losses_f64.npz',
    train_loss=np.array(train_losses), valid_loss=np.array(valid_losses),
)
print('Saved torch float64 mean loss trajectory.')
