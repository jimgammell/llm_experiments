from typing import *
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR as MultiStepLR_PyTorch

class CosineDecayLRSched(LambdaLR):
    def __init__(self, optimizer: optim.Optimizer, total_steps: int, warmup_prop: float = 0.0, const_prop: float = 0.0, min_lr_prop: float = 0.0):
        self.total_steps = total_steps
        self.warmup_prop = warmup_prop
        self.const_prop = const_prop
        self.min_lr_prop = min_lr_prop
        self.warmup_steps = int(self.warmup_prop*self.total_steps)
        self.const_steps = int(self.const_prop*self.total_steps)
        self.decay_steps = self.total_steps - self.warmup_steps - self.const_steps
        self.schedule = np.concatenate([
            np.linspace(0, 1, self.warmup_steps),
            np.ones(self.const_steps),
            self.min_lr_prop + (1.0 - self.min_lr_prop)*(0.5*np.cos(np.linspace(0, np.pi, self.decay_steps)) + 0.5)
        ])
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, current_step):
        assert 0 <= current_step
        if current_step < self.total_steps:
            return self.schedule[current_step]
        else:
            return self.schedule[-1]

class MultiStepLRSched(MultiStepLR_PyTorch):
    def __init__(self, optimizer: optim.Optimizer, total_steps: int, gamma: 0.1, milestone_props: Sequence[float]):
        self.total_steps = total_steps
        self.milestone_props = milestone_props
        super().__init__(
            optimizer=optimizer,
            milestones=[int(milestone_prop*self.total_steps) for milestone_prop in self.milestone_props],
            gamma=gamma
        )