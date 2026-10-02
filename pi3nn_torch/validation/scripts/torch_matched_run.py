"""
Loads the TF side's exact initial weights + preprocessed data
(tf_init_state.npz) into the PyTorch port's networks, then runs a fixed
number of full-batch iterations with early stopping disabled and Adam's
epsilon matched to TF's legacy-Adam default (1e-7, vs PyTorch's 1e-8
default) so the only remaining source of divergence between the two
frameworks is low-level float arithmetic order, not differing starting
points or differing optimizer hyperparameters.

This isolates the question "does the port's forward/backward math agree
with the TF original" from "do two independently-randomly-initialized,
independently-early-stopped training runs land on the same number" --
the comparison already done (run_boston_comparison.py vs TF reference)
answers the second question; this script answers the first.
"""
import sys
sys.path.insert(0, '/home/i0l/hydrogen-emulator-configurable')

import numpy as np
import torch
import torch.nn as nn

from pi3nn_torch.networks import UQNetMean, UQNetStd

DATA = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_init_state.npz')

xTrain = torch.as_tensor(DATA['xTrain'], dtype=torch.float32)
yTrain = torch.as_tensor(DATA['yTrain'], dtype=torch.float32).reshape(-1, 1)
xValid = torch.as_tensor(DATA['xValid'], dtype=torch.float32)
yValid = torch.as_tensor(DATA['yValid'], dtype=torch.float32).reshape(-1, 1)


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

# Sanity check: forward pass on first training row should match TF's
# forward pass on the same row with the same freshly-loaded weights.
with torch.no_grad():
    print('mean(x0):', net_mean(xTrain[:1]).item())
    print('up(x0):  ', net_up(xTrain[:1]).item())
    print('down(x0):', net_down(xTrain[:1]).item())


def train_fixed(model, x_train, y_train, x_valid, y_valid, lr, max_iter, decay_steps, decay_rate, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)  # match TF legacy Adam default eps
    gamma = decay_rate ** (1.0 / decay_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn = nn.MSELoss()

    train_losses = []
    valid_losses = []
    for it in range(max_iter):
        model.train()
        optimizer.zero_grad()
        train_pred = model(x_train)
        train_loss = loss_fn(train_pred, y_train) + model.regularizer_loss()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_pred = model(x_valid)
            valid_loss = loss_fn(valid_pred, y_valid).item()

        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss)
        if it % 100 == 0:
            print(f'[{label}] iter {it}, train_loss {train_loss.item():.6e}, valid_loss {valid_loss:.6e}')

    return train_losses, valid_losses


MAX_ITER = 1200
DECAY_STEPS = 3000
DECAY_RATE = 0.9
LR = 0.02

print('--- MEAN (matched init, no early stop, 1200 fixed iters) ---')
tl, vl = train_fixed(net_mean, xTrain, yTrain, xValid, yValid, LR, MAX_ITER, DECAY_STEPS, DECAY_RATE, 'mean')

np.savez(
    '/home/i0l/UQnet/pi3nn_torch/validation/artifacts/torch_matched_mean_losses.npz',
    train_loss=np.array(tl), valid_loss=np.array(vl),
)
print('Saved torch mean loss trajectory.')
