import os
from copy import copy
from typing import *

from common import *
import datasets
from training_modules.supervised import SupervisedTrainer

class Trial:
    def __init__(self, dataset_name: str, save_dir: Optional[Union[str, os.PathLike]] = None, seed_count: int = 1, module_kwargs: dict = {}, max_epochs: int = 100):
        self.dataset_name = dataset_name
        self.save_dir = os.path.join(OUTPUT_DIR, save_dir if save_dir is not None else dataset_name)
        self.seed_count = seed_count
        self.module_kwargs = module_kwargs
        self.max_epochs = max_epochs
        self.train_dataset, self.test_dataset = datasets.load_dataset(self.dataset_name)
        self.final_run_dir = os.path.join(self.save_dir, 'final_run')
        self.hparam_sweep_dir = os.path.join(self.save_dir, 'hparam_sweep')
        self.optimal_settings = {}
    
    def do_final_run(self):
        for seed in range(self.seed_count):
            subdir = os.path.join(self.final_run_dir, f'seed={seed}')
            os.makedirs(subdir, exist_ok=True)
            base_module_kwargs = copy(self.module_kwargs)
            base_module_kwargs.update(self.optimal_settings)
            trainer = SupervisedTrainer(self.train_dataset, self.test_dataset, base_module_kwargs=base_module_kwargs)
            trainer.run(subdir, max_epochs=self.max_epochs)
    
    def run_hparam_sweep(self):
        experiment_dir = os.path.join(self.hparam_sweep_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        trainer = SupervisedTrainer(self.train_dataset, self.test_dataset, base_module_kwargs=self.module_kwargs)
        self.optimal_settings = trainer.hparam_tune(experiment_dir, max_epochs=self.max_epochs)
    
    def __call__(self, hparam_sweep: bool = False):
        if hparam_sweep:
            self.run_hparam_sweep()
        self.do_final_run()