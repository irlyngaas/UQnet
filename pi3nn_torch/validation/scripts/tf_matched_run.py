"""
Mirrors torch_matched_run.py on the TF side: rebuilds net_mean with the
SAME weights saved in tf_init_state.npz (so this is not relying on RNG
reproducibility a second time, just reloading the exact arrays), trains
with early stopping disabled for a fixed 1200 iterations on the same
data, and saves the per-iteration train/valid loss trajectory so it can
be diffed directly against the PyTorch port's trajectory from the same
starting point.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
import sys
sys.path.insert(0, '/home/i0l/UQnet')

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from pi3nn.Networks.networks import UQ_Net_mean_TF2

DATA = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_init_state.npz')

xTrain = DATA['xTrain'].astype(np.float32)
yTrain = DATA['yTrain'].astype(np.float32)
xValid = DATA['xValid'].astype(np.float32)
yValid = DATA['yValid'].astype(np.float32)

configs = {'num_neurons_mean': [50]}
net_mean = UQ_Net_mean_TF2(configs, 13, 1)
_ = net_mean(xTrain[:1])  # build

net_mean.inputLayer.kernel.assign(DATA['mean_input_kernel'])
net_mean.inputLayer.bias.assign(DATA['mean_input_bias'])
net_mean.fcs[0].kernel.assign(DATA['mean_fc0_kernel'])
net_mean.fcs[0].bias.assign(DATA['mean_fc0_bias'])
net_mean.outputLayer.kernel.assign(DATA['mean_output_kernel'])
net_mean.outputLayer.bias.assign(DATA['mean_output_bias'])

print('mean(x0):', net_mean(xTrain[:1]).numpy().flatten())

criterion = tf.keras.losses.MeanSquaredError()


def add_model_regularizer_loss(model):
    loss = 0
    for l in model.layers:
        if hasattr(l, 'kernel_regularizer') and l.kernel_regularizer:
            loss += l.kernel_regularizer(l.kernel)
        if hasattr(l, 'bias_regularizer') and l.bias_regularizer:
            loss += l.bias_regularizer(l.bias)
    return loss


LR = 0.02
MAX_ITER = 1200
DECAY_STEPS = 3000
DECAY_RATE = 0.9

global_step = tf.Variable(0, trainable=False)
decayed_lr = tf.compat.v1.train.exponential_decay(LR, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE, staircase=False)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=decayed_lr)

train_losses = []
valid_losses = []
for it in range(MAX_ITER):
    with tf.GradientTape() as tape:
        train_pred = net_mean(xTrain, training=True)
        train_loss = criterion(yTrain, train_pred)
        train_loss += add_model_regularizer_loss(net_mean)
    gradients = tape.gradient(train_loss, net_mean.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net_mean.trainable_variables))
    global_step.assign_add(1)

    valid_pred = net_mean(xValid, training=False)
    valid_loss = criterion(yValid, valid_pred)

    train_losses.append(float(train_loss.numpy()))
    valid_losses.append(float(valid_loss.numpy()))
    if it % 100 == 0:
        print(f'[mean] iter {it}, train_loss {train_loss.numpy():.6e}, valid_loss {valid_loss.numpy():.6e}')

np.savez(
    '/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_matched_mean_losses.npz',
    train_loss=np.array(train_losses), valid_loss=np.array(valid_losses),
)
print('Saved TF mean loss trajectory.')
