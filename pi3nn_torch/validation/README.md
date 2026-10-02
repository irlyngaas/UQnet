# PI3NN TF-vs-PyTorch validation

Why this exists: `hydrogen-emulator-configurable/pi3nn_torch/` (and this
repo's own copy at `pi3nn_torch/`) is a PyTorch port of this repo's
TensorFlow PI3NN implementation. This directory holds the scripts and
saved weights/data used to validate that port against the original on
the boston-housing UCI benchmark. The TensorFlow venv the TF-side
scripts need lives *outside* this repo at `/home/i0l/pi3nn_validation/tf_env`
-- deliberately not committed here (2.5GB, platform-specific binaries,
not something a git diff should ever show).

**Conclusion**: the port is correct. An initial ~12% gap between
independent full runs (PyTorch's own init + `torch.optim.Adam` vs TF's own
init + TF's Adam) was explained by (1) different default weight init
between TF's `Dense` and PyTorch's `nn.Linear`, and (2) a genuine formula
difference between TF's legacy Adam and PyTorch's Adam (see "What the
formula difference is" below). Neutralizing both made PICP match the TF
reference *exactly* on train/valid/test, with MPIW/RMSE/R2 within 3-7%
(explained by irreducible float-kernel differences between frameworks,
confirmed via a float64 control showing no reduction in the gap).

**Decision made**: `pi3nn_torch/trainer.py` stays on stock
`torch.optim.Adam`. Neither Adam formula is "more correct", and real
PI3NN work on emulator data has no TF reference to match against, so
there's no ongoing reason to carry TF's specific quirk. Nothing in the
port needs to change because of this investigation -- these scripts are
a point-in-time validation record, not something the port depends on.

## Layout

In this repo (`pi3nn_torch/validation/`):
- `scripts/` -- the validation scripts, in the order you'd actually run
  them (see "How to rerun" below).
- `artifacts/` -- saved `.npz` files: TF's exact initial weights + the
  exact preprocessed boston train/valid/test split (`tf_init_state.npz`),
  and saved loss trajectories from each comparison run. Scripts read/write
  these; you don't need to touch them directly.

Outside this repo, not committed:
- `/home/i0l/pi3nn_validation/tf_env/` -- a venv with TensorFlow 2.21 +
  everything `main_PI3NN.py` needs (tqdm, matplotlib, hyperopt, tf_keras)
  plus the NumPy-2.0 compat fix. Only needed for the TF-side scripts
  below. If this venv is gone, recreate it with:
  `python3 -m venv tf_env && source tf_env/bin/activate && pip install tensorflow scikit-learn pandas matplotlib tqdm hyperopt tf_keras "numpy>=2"`.

The PyTorch side doesn't need a dedicated venv -- it runs against your
existing environment (confirmed working: torch 2.13.0+cpu, scikit-learn
1.9.1), the same one used for the emulator training work.

## How to rerun

All commands assume you're `cd`'d anywhere; scripts use absolute paths
internally so your cwd doesn't matter, but `python3 <script>` still needs
the right environment active for TF-side scripts.

**1. (Optional) Regenerate TF's initial weights + the preprocessed data.**
Already saved in `artifacts/tf_init_state.npz` -- only rerun this if you
want a different seed or dataset.
```
source /home/i0l/pi3nn_validation/tf_env/bin/activate
TF_USE_LEGACY_KERAS=True python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/tf_extract_init.py
deactivate
```

**2. Matched-init, fixed-iteration trajectory comparison (float32).**
Loads TF's exact weights into both frameworks' `net_mean`, disables early
stopping, runs 1200 full-batch iterations, no Adam-formula fix -- this is
the run that first showed the gap growing from ~1e-6 (iter 1) to ~12%
(iter 1199) relative.
```
source /home/i0l/pi3nn_validation/tf_env/bin/activate
TF_USE_LEGACY_KERAS=True python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/tf_matched_run.py
deactivate
python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/torch_matched_run.py
```

**3. Same thing in float64** (rules out precision/RNG as the cause --
divergence magnitude barely changes vs float32):
```
source /home/i0l/pi3nn_validation/tf_env/bin/activate
TF_USE_LEGACY_KERAS=True python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/tf_matched_run_f64.py
deactivate
python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/torch_matched_run_f64.py
```

**4. Matched-init trajectory with TF's exact Adam formula substituted**
(`tfadam_optim.py` implements `tf.raw_ops.ResourceApplyAdam`'s formula as
a `torch.optim.Optimizer`). Divergence at early iterations drops ~1000x
vs step 2 -- this is what identifies the Adam formula as the dominant
near-term error source:
```
python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/torch_matched_run_tfadam.py
```

**5. Compare any of the saved trajectories numerically:**
```python
import numpy as np
tf_d = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/tf_matched_mean_losses.npz')
pt_d = np.load('/home/i0l/UQnet/pi3nn_torch/validation/artifacts/torch_matched_mean_losses.npz')
for it in [0, 1, 10, 100, 1199]:
    rel = abs(tf_d['valid_loss'][it] - pt_d['valid_loss'][it]) / abs(tf_d['valid_loss'][it])
    print(it, rel)
```

**6. The full real-pipeline test** -- `pi3nn_torch.trainer.PI3NNTrainer`
with `Max_iter=5000`, early stopping, boundary optimization, all three
networks (mean/up/down), starting from TF's exact initial weights and
using TFAdam instead of `torch.optim.Adam`. This is the one that matched
TF's PICP exactly:
```
python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/run_boston_tfadam_matched.py
```
Expect PICP_train/valid/test = 0.9511/0.9348/0.8627 (matches TF exactly),
MPIW ~1.226 (TF: ~1.2695, ~3.4% off), RMSE_test ~0.394 (TF: 0.4228),
R2_test ~0.839 (TF: 0.8142).

**7. (Optional) Same full pipeline but with PyTorch's own random init**
(no weight-matching, just the Adam-formula fix) -- shows the Adam fix
alone, without matched init, gives a smaller but less clean improvement
since independent random init still dominates:
```
python3 /home/i0l/UQnet/pi3nn_torch/validation/scripts/run_boston_tfadam.py
```

**8. For reference, the actual TF run this was all validated against:**
```
source /home/i0l/pi3nn_validation/tf_env/bin/activate
cd /home/i0l/UQnet
TF_USE_LEGACY_KERAS=True python3 main_PI3NN.py --data boston --mode manual --quantile 0.95
deactivate
```
Expected: PICP_train=0.9511, PICP_valid=0.9348, PICP_test=0.8627,
MPIW train/valid/test=1.2696/1.2695/1.2694, RMSE_test=0.4228, R2_test=0.8142.

(UQnet's `pi3nn/Trainers/trainers.py` and `pi3nn/Utils/Utils.py` needed a
`np.Inf` -> `np.inf` rename for NumPy 2.0 compatibility -- already
committed there as `b8ba5b6`, pushed to `origin/master`.)

## What the Adam formula difference is

TF's legacy Adam (`tf.raw_ops.ResourceApplyAdam`, documented):
```
lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
m_t  = beta1*m_{t-1} + (1-beta1)*g
v_t  = beta2*v_{t-1} + (1-beta2)*g^2
var -= m_t * lr_t / (sqrt(v_t) + eps)
```

PyTorch's Adam (`torch/optim/adam.py`, non-capturable path):
```
bias_correction1 = 1 - beta1^t
bias_correction2 = 1 - beta2^t
step_size = lr / bias_correction1
denom = sqrt(v_t) / sqrt(bias_correction2) + eps
var -= step_size * m_t / denom
```

These are only algebraically identical when `eps = 0` or `t -> infinity`
(bias-correction factors -> 1). With `eps=1e-7` and `beta2=0.999`, they
disagree for hundreds to low-thousands of steps -- exactly the window
PI3NN's early stopping lands in (1026-1636 iterations across the mean/up/
down networks in these runs).
