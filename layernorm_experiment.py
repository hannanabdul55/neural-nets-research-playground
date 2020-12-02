import numpy as np
import torch
import pandas as pd
from utils import *
import torchvision.datasets as datasets
import torchvision
from time import time
import vgg as vgg_test

import torch
import torch.nn as nn
from scale_layer import ScaleLayer
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from lit_model import *
import torchvision.models as models

import argparse
import json
import ray
import copy

parser = argparse.ArgumentParser(description='Process some config values')
parser.add_argument("--config")
parser.add_argument("checkpoint", nargs='?', default='results-' + str(time()))
parser.add_argument("--threads")
parser.add_argument("--dir")
parser.add_argument("--gpus")
parser.add_argument("--workers")
parser.add_argument("--local")

args = parser.parse_args()
if args.local:
    use_local = True
else:
    use_local = False
if args.dir:
    dir = args.dir
else:
    dir = 'results_default'

kwargs = {
    'local_mode': use_local
}
if args.gpus:
    n_gpus = int(args.gpus)
    kwargs['num_gpus'] = n_gpus
else:
    n_gpus = 0

if args.workers:
    workers = int(args.workers)
else:
    workers = 5

if args.config:
    exp_config = json.load(open(args.config, "r"))
else:
    exp_config = {
        'name': 'first_run',
        'exps': [
        # {
        #     'name': 'layernorm',
        #     'norm_type': 'layernorm'
        # },
        {
            'name': 'maxnorm',
            'norm_type': 'maxnorm'
        },
        {
            'name': 'batchnorm',
            'norm_type': 'batchnorm'
        },
        {
            'name': 'nonorm',
            'epochs': 50
        }
    ]
    }

@ray.remote(num_gpus=1)
def run_experiment(exp):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=10)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=10)
    if 'norm_type' in exp:
        if exp['norm_type'] == 'layernorm':
            norm = nn.LayerNorm
        elif exp['norm_type'] == 'maxnorm':
            norm = ScaleLayer
        else:
            # exp['norm_type'] == 'batchnorm':
            norm = nn.BatchNorm2d
    else:
        norm = None

    if 'epochs' in exp:
        e = exp['epochs']
    else:
        e = 50

    res = {
        'config': exp
    }
    model_type = 'vgg'
    if 'model' in exp:
        model_type = exp['model']
    if model_type == 'resnet':
        model = torchvision.models.resnet50(pretrained=False, norm_layer=norm)
    else:
        if norm is None:
            model = vgg_test.vgg16()
        else:
            model = vgg_test.vgg11_bn(norm_layer=norm)

    trainer = pl.Trainer(gpus=1,max_epochs=e, progress_bar_refresh_rate=10)

    trainer.fit(LitModel(model, name = exp['name']), train_loader, val_loader)

    pass


if __name__ == '__main__':

    if 'name' in exp_config:
        dir = f"result/result_{exp_config['name']}"
    os.makedirs(dir, exist_ok=True)
    n_test = 1e6 * 6
    has_gpu = torch.cuda.is_available()
    print(f"GPU ? {has_gpu}")
    if has_gpu:
        print(f"Initializing ray with {2} GPUs")
        print('Available devices ', torch.cuda.device_count())
        ray.init(num_gpus= torch.cuda.device_count())
    else:
        print(f"Initializing with no GPUs")
        ray.init()

    a = time()
    futures = [run_experiment.remote(x) for x in exp_config['exps']]
    res = ray.get(futures)
    b = time()
    print(f"Experiment {exp_config['name']} ran in {b-a} seconds")


