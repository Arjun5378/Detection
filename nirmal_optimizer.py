import torch
from torch.optim.optimizer import Optimizer

class NirmalOptimizer(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9): 
        defaults = dict(lr=lr, beta=beta)
        super(NirmalOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    state['momentum_buffer'].mul_(beta).add_(d_p, alpha=1 - beta)

                p.add_(state['momentum_buffer'], alpha=-lr)

        return loss
