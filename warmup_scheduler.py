from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ConstantWarmupScheduler(LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer, 
                 lr_scheduler: LRScheduler, 
                 warmup_epoch: int, 
                 warmup_lr: float, 
                 last_epoch: int=-1, 
                 verbose="deprecated"):
        self.lr_scheduler = lr_scheduler
        self.warmup_epoch = warmup_epoch
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.lr_scheduler.get_last_lr()
        else:
            return [self.warmup_lr for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.lr_scheduler.step()
        super().step(epoch)



class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_scheduler: LRScheduler, warmup_epoch: int, min_lr: float, last_epoch: int=-1, verbose="deprecated"):
        self.lr_scheduler = lr_scheduler
        self.warmup_epoch = warmup_epoch
        self.min_lr = min_lr
        self.max_lr = self.lr_scheduler.get_last_lr()[0]
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.lr_scheduler.get_last_lr()
        else:
            current_lr = (self.max_lr - self.min_lr) * self.last_epoch / (self.warmup_epoch - 1) + self.min_lr
            return [current_lr for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.lr_scheduler.step()
        super().step(epoch)