"""
Same matched-init comparison as torch_matched_run.py, but replaces
torch.optim.Adam with a hand-rolled optimizer that implements TF's
documented tf.raw_ops.ResourceApplyAdam update rule exactly:

    lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
    m_t  = beta1*m_{t-1} + (1-beta1)*g
    v_t  = beta2*v_{t-1} + (1-beta2)*g^2
    var -= m_t * lr_t / (sqrt(v_t) + eps)

vs PyTorch's documented Adam update rule:

    bias_correction1 = 1 - beta1^t
    bias_correction2 = 1 - beta2^t
    step_size = lr / bias_correction1
    denom = sqrt(v_t) / sqrt(bias_correction2) + eps
    var -= step_size * m_t / denom

These are NOT algebraically identical except in the eps=0 or t->inf limit.
This script tests whether using TF's exact formula instead of PyTorch's
built-in Adam collapses the divergence seen in torch_matched_run.py.
"""
import sys
sys.path.insert(0, '/home/i0l/hydrogen-emulator-configurable')

import numpy as np
import torch
import torch.nn as nn

from pi3nn_torch.networks import UQNetMean

DATA = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_init_state.npz')

xTrain = torch.as_tensor(DATA['xTrain'], dtype=torch.float32)
yTrain = torch.as_tensor(DATA['yTrain'], dtype=torch.float32).reshape(-1, 1)
xValid = torch.as_tensor(DATA['xValid'], dtype=torch.float32)
yValid = torch.as_tensor(DATA['yValid'], dtype=torch.float32).reshape(-1, 1)

net_mean = UQNetMean(13, 1, num_neurons=[50])
with torch.no_grad():
    net_mean.input_layer.weight.copy_(torch.as_tensor(DATA['mean_input_kernel'].T))
    net_mean.input_layer.bias.copy_(torch.as_tensor(DATA['mean_input_bias']))
    net_mean.fcs[0].weight.copy_(torch.as_tensor(DATA['mean_fc0_kernel'].T))
    net_mean.fcs[0].bias.copy_(torch.as_tensor(DATA['mean_fc0_bias']))
    net_mean.output_layer.weight.copy_(torch.as_tensor(DATA['mean_output_kernel'].T))
    net_mean.output_layer.bias.copy_(torch.as_tensor(DATA['mean_output_bias']))

print('mean(x0):', net_mean(xTrain[:1]).item())

LR0 = 0.02
MAX_ITER = 1200
DECAY_STEPS = 3000
DECAY_RATE = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-7

loss_fn = nn.MSELoss()
params = list(net_mean.parameters())
m = [torch.zeros_like(p) for p in params]
v = [torch.zeros_like(p) for p in params]

train_losses = []
valid_losses = []
for it in range(MAX_ITER):
    # TF's exponential_decay(staircase=False) uses global_step BEFORE
    # increment for this step's lr, matching main loop semantics.
    lr_decayed = LR0 * (DECAY_RATE ** (it / DECAY_STEPS))
    beta1_power = BETA1 ** (it + 1)
    beta2_power = BETA2 ** (it + 1)
    lr_t = lr_decayed * (1 - beta2_power) ** 0.5 / (1 - beta1_power)

    net_mean.train()
    for p in params:
        if p.grad is not None:
            p.grad = None
    train_pred = net_mean(xTrain)
    train_loss = loss_fn(train_pred, yTrain) + net_mean.regularizer_loss()
    train_loss.backward()

    with torch.no_grad():
        for i, p in enumerate(params):
            g = p.grad
            m[i].mul_(BETA1).add_(g, alpha=1 - BETA1)
            v[i].mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
            p.sub_(lr_t * m[i] / (v[i].sqrt() + EPS))

    net_mean.eval()
    with torch.no_grad():
        valid_pred = net_mean(xValid)
        valid_loss = loss_fn(valid_pred, yValid).item()

    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss)
    if it % 100 == 0:
        print(f'[mean-tfadam] iter {it}, train_loss {train_loss.item():.10e}, valid_loss {valid_loss:.10e}')

np.savez(
    '/home/i0l/UQnet/pi3nn_torch/validation/artifacts/torch_matched_mean_losses_tfadam.npz',
    train_loss=np.array(train_losses), valid_loss=np.array(valid_losses),
)
print('Saved torch (TF-formula Adam) mean loss trajectory.')
