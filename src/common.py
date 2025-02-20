import os
import socket
import yaml
import torch

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'outputs')
CONFIG_DIR = os.path.join(PROJ_DIR, 'config')
RESOURCE_DIR = os.path.join(PROJ_DIR, 'resources')

with open(os.path.join(CONFIG_DIR, 'per_machine_config.yaml'), 'r') as f:
    per_machine_configs = yaml.load(f, Loader=yaml.FullLoader)
hostname = socket.gethostname()
host_config = None
for key in per_machine_configs.keys():
    if per_machine_configs[key]['hostname_contains'] in hostname:
        assert host_config is None
        host_config = per_machine_configs[key]
assert host_config is not None
dataset_paths = host_config['dataset_paths']
MNIST_DIR = dataset_paths['mnist'] if ('mnist' in dataset_paths) and (dataset_paths['mnist'] is not None) else os.path.join(RESOURCE_DIR, 'mnist')
CIFAR10_DIR = dataset_paths['cifar10'] if ('cifar10' in dataset_paths) and (dataset_paths['cifar10'] is not None) else os.path.join(RESOURCE_DIR, 'cifar10')
IMAGENET_DIR = dataset_paths['imagenet']
REDPAJAMA_DIR = dataset_paths['redpajama']
REDPAJAMA_SAMPLE_DIR = dataset_paths['redpajama_sample']

COMPILE_MODELS = host_config['compile_models']

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = 10*gpu_properties.major + gpu_properties.minor
    if arch >= 70:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

PLOT_WIDTH = 4
PLOT_KWARGS = {'rasterized': True}
SAVEFIG_KWARGS = {'dpi': 300}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESOURCE_DIR, exist_ok=True)