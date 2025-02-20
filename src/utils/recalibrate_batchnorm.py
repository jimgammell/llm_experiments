from typing import *
import torch

@torch.no_grad()
def recalibrate_batchnorm(trainer, module):
    model = module.classifier
    device = module.device
    dataloader = trainer.datamodule.train_dataloader(override_batch_size=trainer.datamodule.eval_batch_size)
    training_mode = model.training
    model.eval()
    for mod in model.modules():
        if not isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            continue
        mod.reset_running_stats()
        mod.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        _ = module.step((x, y), train=False)
    module.train(training_mode)