"""
torch.optim.Optimizer implementing TF's documented tf.raw_ops.ResourceApplyAdam
update exactly (see conversation): lr_t = lr*sqrt(1-beta2^t)/(1-beta1^t);
var -= m_t * lr_t / (sqrt(v_t) + eps). Used only to test how much of the
PyTorch-port vs TF-reference gap on the full boston benchmark is explained
by this formula difference vs irreducible float noise.
"""
import torch
from torch.optim import Optimizer


class TFAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                state['step'] += 1
                t = state['step']
                m, v = state['m'], state['v']
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                beta1_power = beta1 ** t
                beta2_power = beta2 ** t
                lr_t = lr * (1 - beta2_power) ** 0.5 / (1 - beta1_power)
                p.sub_(lr_t * m / (v.sqrt() + eps))
        return loss
