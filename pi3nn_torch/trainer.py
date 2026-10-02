"""
Port of UQnet's pi3nn/Trainers/trainers.py (CL_trainer), non-batch
(full-gradient-descent) path only -- matches what main_PI3NN.py's manual
mode actually exercises for the UCI benchmarks (batch_training=False).
Batch-mode training, hyperopt auto-mode, and the plotting/saving utilities
from the original are intentionally not ported here; this is scoped to
reproducing the core algorithm for a correctness check, not the full
feature set.

Sequencing matters and is preserved exactly: net_mean trains to completion
first; only then are its residuals computed and split by sign into
disjoint subsets; net_up and net_down each train to completion on their
own subset, never seeing the original y or each other's data.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .boundary_optimizer import BoundaryOptimizer
from .distributed import is_main_process


def _make_optimizer(name, params, lr):
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr)
    elif name == 'SGD':
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f'Optimizer {name} not supported')


class PI3NNTrainer:
    def __init__(
        self, configs, net_mean, net_up, net_down,
        x_train, y_train, x_valid=None, y_valid=None, x_test=None, y_test=None,
        device='cpu',
    ):
        self.configs = configs
        self.device = device
        self.net_mean = net_mean.to(device)
        self.net_up = net_up.to(device)
        self.net_down = net_down.to(device)

        self.x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        self.y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
        self.x_valid = torch.as_tensor(x_valid, dtype=torch.float32, device=device) if x_valid is not None else None
        self.y_valid = torch.as_tensor(y_valid, dtype=torch.float32, device=device).reshape(-1, 1) if y_valid is not None else None
        self.x_test = torch.as_tensor(x_test, dtype=torch.float32, device=device) if x_test is not None else None
        self.y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device).reshape(-1, 1) if y_test is not None else None

        # DistributedDataParallel only ever wraps the forward pass used for
        # the gradient step (see _train_one_network) -- boundary_optimization
        # and evaluate() always call self.net_mean/up/down (the raw,
        # unwrapped modules) directly, since those are forward-only,
        # full-dataset computations that every rank can redundantly repeat
        # without needing gradient synchronization at all.
        self._ddp_mean = self._ddp_up = self._ddp_down = None
        if dist.is_initialized():
            device_ids = [torch.device(device).index] if torch.device(device).type == 'cuda' else None
            self._ddp_mean = DDP(self.net_mean, device_ids=device_ids)
            self._ddp_up = DDP(self.net_up, device_ids=device_ids)
            self._ddp_down = DDP(self.net_down, device_ids=device_ids)

    def _local_shard(self, x, y):
        """
        Splits x/y's rows evenly across ranks (drop_last: truncates to a
        multiple of world_size first) so every rank's local shard is
        exactly the same size. DDP averages gradients uniformly across
        ranks, not weighted by each rank's local batch size, so unequal
        shards would silently bias the effective gradient away from the
        true full-batch mean this was validated against on CPU.
        """
        if not dist.is_initialized():
            return x, y
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        n_local = x.shape[0] // world_size
        start = rank * n_local
        end = start + n_local
        return x[start:end], y[start:end]

    def _train_one_network(self, model, x_train, y_train, x_valid, y_valid, lr, optimizer_name, label, ddp_model=None):
        cfg = self.configs
        forward_model = ddp_model if ddp_model is not None else model
        optimizer = _make_optimizer(optimizer_name, model.parameters(), lr)
        scheduler = None
        if cfg.get('exponential_decay', False):
            gamma = cfg['decay_rate'] ** (1.0 / cfg['decay_steps'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        loss_fn = nn.MSELoss()
        best_loss = math.inf
        best_state = None
        wait = 0

        x_train_local, y_train_local = self._local_shard(x_train, y_train)

        for it in range(cfg['Max_iter']):
            model.train()
            optimizer.zero_grad()
            train_pred = forward_model(x_train_local)
            train_loss = loss_fn(train_pred, y_train_local) + model.regularizer_loss()
            train_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                valid_pred = model(x_valid)
                valid_loss = loss_fn(valid_pred, y_valid).item()

            if it % 100 == 0 and cfg.get('verbose', 0) > 0 and is_main_process():
                print(f'[{label}] iter {it}, train_loss {train_loss.item():.4e}, valid_loss {valid_loss:.4e}')

            if cfg.get('early_stop', False) and it >= cfg['early_stop_start_iter']:
                if dist.is_initialized():
                    # Every rank's weights stay value-identical after each
                    # optimizer.step() (DDP averages gradients before the
                    # update), so in principle every rank would reach the
                    # same early-stop decision on its own -- but that
                    # relies on bit-identical floating point across GPUs,
                    # which isn't guaranteed. Broadcasting rank 0's
                    # decision instead removes that risk entirely: all
                    # ranks are forced into lockstep regardless of any
                    # per-GPU numeric noise.
                    if dist.get_rank() == 0:
                        if valid_loss < best_loss:
                            new_best_loss, new_wait, improved = valid_loss, 0, 1.0
                        else:
                            new_best_loss, new_wait, improved = best_loss, wait + 1, 0.0
                        do_stop = 1.0 if new_wait >= cfg['wait_patience'] else 0.0
                    else:
                        new_best_loss, new_wait, improved, do_stop = 0.0, 0.0, 0.0, 0.0
                    state = torch.tensor(
                        [new_best_loss, float(new_wait), improved, do_stop],
                        dtype=torch.float64, device=self.device,
                    )
                    dist.broadcast(state, src=0)
                    best_loss, wait_f, improved, do_stop = state.tolist()
                    wait = int(wait_f)
                    if improved and cfg.get('restore_best_weights', False):
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    if do_stop:
                        if cfg.get('restore_best_weights', False) and best_state is not None:
                            model.load_state_dict(best_state)
                        if cfg.get('verbose', 0) > 0 and is_main_process():
                            print(f'[{label}] early stop at iter {it}')
                        break
                else:
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        wait = 0
                        if cfg.get('restore_best_weights', False):
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= cfg['wait_patience']:
                            if cfg.get('restore_best_weights', False) and best_state is not None:
                                model.load_state_dict(best_state)
                            if cfg.get('verbose', 0) > 0:
                                print(f'[{label}] early stop at iter {it}')
                            break

    def train(self):
        cfg = self.configs

        self._train_one_network(
            self.net_mean, self.x_train, self.y_train, self.x_valid, self.y_valid,
            lr=cfg['lr'][0], optimizer_name=cfg['optimizers'][0], label='mean',
            ddp_model=self._ddp_mean,
        )

        # Only after net_mean is fully trained: compute residuals and split
        # by sign into disjoint subsets. net_up/net_down never see y_train
        # directly, only these one-sided residual magnitudes.
        self.net_mean.eval()
        with torch.no_grad():
            diff_train = self.y_train - self.net_mean(self.x_train)
            diff_valid = self.y_valid - self.net_mean(self.x_valid)

        up_mask_train = (diff_train > 0).squeeze(-1)
        down_mask_train = (diff_train < 0).squeeze(-1)
        up_mask_valid = (diff_valid > 0).squeeze(-1)
        down_mask_valid = (diff_valid < 0).squeeze(-1)

        x_train_up, y_train_up = self.x_train[up_mask_train], diff_train[up_mask_train]
        x_valid_up, y_valid_up = self.x_valid[up_mask_valid], diff_valid[up_mask_valid]
        x_train_down, y_train_down = self.x_train[down_mask_train], -diff_train[down_mask_train]
        x_valid_down, y_valid_down = self.x_valid[down_mask_valid], -diff_valid[down_mask_valid]

        self._train_one_network(
            self.net_up, x_train_up, y_train_up, x_valid_up, y_valid_up,
            lr=cfg['lr'][1], optimizer_name=cfg['optimizers'][1], label='up',
            ddp_model=self._ddp_up,
        )
        self._train_one_network(
            self.net_down, x_train_down, y_train_down, x_valid_down, y_valid_down,
            lr=cfg['lr'][2], optimizer_name=cfg['optimizers'][2], label='down',
            ddp_model=self._ddp_down,
        )

    def boundary_optimization(self, verbose=0):
        cfg = self.configs
        n_train = self.x_train.shape[0]
        self.net_mean.eval()
        self.net_up.eval()
        self.net_down.eval()
        with torch.no_grad():
            output_mean = self.net_mean(self.x_train)
            output_up = self.net_up(self.x_train)
            output_down = self.net_down(self.x_train)

        num_outlier = int(n_train * (1 - cfg['quantile']) / 2)
        optimizer = BoundaryOptimizer(
            self.y_train, output_mean, output_up, output_down,
            num_outlier=num_outlier,
            c_up0_ini=0.0, c_up1_ini=100000.0,
            c_down0_ini=0.0, c_down1_ini=100000.0,
            max_iter=1000,
        )
        self.c_up = optimizer.optimize_up(verbose=verbose)
        self.c_down = optimizer.optimize_down(verbose=verbose)
        if verbose > 0:
            print(f'c_up: {self.c_up}, c_down: {self.c_down}')

    def evaluate(self, final_evaluation=True, verbose=0):
        """Returns a dict of PICP/MPIW (train/valid/test) + MSE/RMSE/R2 (test)."""
        self.net_mean.eval()
        self.net_up.eval()
        self.net_down.eval()

        def _caps(x, y):
            with torch.no_grad():
                mean = self.net_mean(x).cpu().numpy().flatten()
                up = self.net_up(x).cpu().numpy().flatten()
                down = self.net_down(x).cpu().numpy().flatten()
            y = y.cpu().numpy().flatten()
            upper = mean + self.c_up * up
            lower = mean - self.c_down * down
            picp = float(np.mean((y <= upper) & (y >= lower)))
            mpiw = float(np.mean(upper - lower))
            return picp, mpiw, mean

        results = {}
        results['picp_train'], results['mpiw_train'], _ = _caps(self.x_train, self.y_train)
        results['picp_valid'], results['mpiw_valid'], _ = _caps(self.x_valid, self.y_valid)

        if final_evaluation and self.x_test is not None:
            results['picp_test'], results['mpiw_test'], test_mean = _caps(self.x_test, self.y_test)
            y_test = self.y_test.cpu().numpy().flatten()
            mse = float(np.mean((test_mean - y_test) ** 2))
            results['mse_test'] = mse
            results['rmse_test'] = mse ** 0.5
            ss_res = np.sum((y_test - test_mean) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            results['r2_test'] = float(1 - ss_res / ss_tot)

        if verbose > 0:
            for k, v in results.items():
                print(f'{k}: {v:.4f}')
        return results
