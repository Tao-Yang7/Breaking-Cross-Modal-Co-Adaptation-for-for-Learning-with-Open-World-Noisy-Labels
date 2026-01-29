import argparse
import pdb
import sys
import csv
import datetime
import importlib
import os
import time
import torch, time, gc
from collections import defaultdict
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import wandb
import open_clip.transformer
import torchvision
from torchvision import datasets, transforms

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from core import my_test_nega_clip, training_main_loop, analyze_sample_filtering, my_test_lora
from core import test_split_prompt, test_split_lora
from core.train_delora import deora_training_main_loop
from core.filter_top_k_samples import *
from models import scheduler_builder
from models.models import NegaPromptCLIP, OriginalCLIP
from utils.ema import EMA, attach_ema_to_optimizer

from tqdm import tqdm
import numpy as np
from scipy import interpolate
from sklearn import metrics
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import roc_auc_score as Auc
from sklearn.metrics import roc_curve as Roc


from data.stanford.my_stanford_cars import *
import os

os.environ["WANDB_MODE"] = "disabled"

_tokenizer = _Tokenizer()
parser = argparse.ArgumentParser("Training")

# Distribute
parser.add_argument("--local_rank", type=int)

# Dataset
parser.add_argument('--dataset', type=str, default='cifar100',
                    help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | ImageNet_p[1-10]| OOD_ImageNet_[SUN|iNaturalist|places365|dtd")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--closeset-ratio', type=float, default=0.2)

# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)

# misc
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--use-cpu', action='store_true', default=False)

# clip
parser.add_argument('--clip_backbone', type=str,
                    default='ViT-B/16')  # RN50 RN101 RN50x4 RN50x16 RN50x64 ViT-B/32 ViT-B/16 ViT-B/14 ViT-L/14@336px
parser.add_argument('--CSC', type=int, default=0)
parser.add_argument('--LOG', type=int, default=1)
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--positive_pth', type=str, default='')  # xxxx.pth
parser.add_argument('--negative_pth', type=str, default='')  # xxxx.pth
parser.add_argument('--NEGA_CTX', type=int, default=2)

# POMP
parser.add_argument('--POMP', type=int, default=0)
parser.add_argument('--new', action='store_true', default=False)

# LoRA配置：仅将 PEFT 应用到最后 N 层
parser.add_argument('--delora', action='store_true', help='Enable DeLoRA training (optional)')
parser.add_argument('--partial', type=int, default=1, help='Number of top transformer layers to apply PEFT (default: all layers)')
parser.add_argument('--rank', type=int, default=2, help='rank of LoRA')
from data.cifar100 import class_names

def print_results(results, indent=0):
    prefix = '  ' * indent
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_results(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def save_nega_prompt_model(model, save_path):
    """
    保存 NegaPromptCLIP 模型，特别处理关键参数
    """
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v.cpu()  # 移动到CPU

    torch.save({
        'model_state_dict': state_dict,
        # 专门保存关键参数，便于直接访问
        'ctx_positive': model.prompt_learner.ctx_positive.cpu(),
        'ctx_negative': model.prompt_learner.ctx_negative.cpu(),
    }, save_path)


def load_nega_prompt_model(model, load_path, device='cuda'):
    """
    专门针对 NegaPromptCLIP 模型的加载
    """
    checkpoint = torch.load(load_path, map_location='cuda', weights_only=False)

    # 加载状态字典
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 移动到目标设备
    model = model.to(device)

    print(f"Model loaded from {load_path}")
    print(f"ctx_positive shape: {model.prompt_learner.ctx_positive.shape}")
    print(f"ctx_negative shape: {model.prompt_learner.ctx_negative.shape}")

    return model


def main_worker(options):
    run = wandb.init(project="few_shot_stage1", dir='.', reinit=True)
    run.config.update(options, allow_val_change=True)
    options = run.config
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    # if options['use_cpu']: use_gpu = False
    print("Currently using GPU: {}".format(options['gpu']))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(options['seed'])


    # Dataset
    print("{} Preparation".format(options['dataset']))
    if args.dataset == "cifar80N":
        from data.cifar100 import build_dataset_loader
        from data.cifar100 import class_names
        _, trainloader, testloader = build_dataset_loader(
            closeset_ratio=args.closeset_ratio,
            batch_size=args.batch_size,
            noise_type='asymmetric'
        )
        classnames = class_names[:80]
    elif args.dataset == "stanford_cars":
        trainloader, testloader = get_stanford_cars(
            batch_size=args.batch_size,
            num_shots=16,
            noise_type='symmetric',
            noise_rate=args.closeset_ratio,
            seed=42
        )
        classnames = get_stanford_cars_classnames()
    elif args.dataset == "crop_cifar100":
        from data.crop_cifar100.mixed_cifar100_imagenet32 import build_dataset_loader, class_names, ID_class_names
        _, trainloader, testloader = build_dataset_loader(512, id_noise_ratio=0.4, ood_noise_ratio=0.4)
        classnames = class_names


    elif args.dataset == "place365":
        from data.place365_cifar100.extended_mixed_dataset import get_cifar100_loaders
        import torchvision.transforms as transforms
        from data.crop_cifar100.mixed_cifar100_imagenet32 import class_names, ID_class_names
        cifar_path = "/data/tyang/aaaa_now/compare/data/cifar100"
        place365_path = "/data/tyang/aaaa_now/compare/data/place365"
        trainloader, testloader = get_cifar100_loaders(
            data_root=cifar_path,
            place365_root=place365_path,
            batch_size=256,
            ood_ratio=0.4,  # 您想要的参数
            id_noise_ratio=0.4,  # 您想要的参数
            num_workers=4
        )
        classnames = class_names
    else:
        from data.webfg import build_webfg_dataloader, get_webfg_class_names
        dataset_name = args.dataset

        # 创建训练集和测试集加载器
        dataset, trainloader, testloader = build_webfg_dataloader(
            dataset_name,
            batch_size=args.batch_size,
            num_workers=8
        )
        classnames = get_webfg_class_names(dataset_name)


    
    options['num_classes'] = len(classnames)
    options['CTX_INIT'] = "a photo of a"
    test_labels = [classname.replace('_', ' ') for classname in classnames]
    options['classnames'] = test_labels

    # Web-aircraft
    # raw_classnames = [classname.replace('_', ' ') for classname in classnames]
    # options['classnames'] = [f"{name} aircraft" for name in raw_classnames]

    options['N_CTX'] = 4
    print("CLIP backbone: {}".format(options['clip_backbone']))
    device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")

    # # server 3
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #     'ViT-B-16',
    #     pretrained='/home/tyang/.cache/huggingface/hub/models--timm--vit_base_patch16_clip_224.openai/snapshots/977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_pytorch_model.bin'
    # )

    # server5
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16',
        pretrained='/data/tyang/aaaa_now/.cache/huggingface/hub/models--timm--vit_base_patch16_clip_224.openai/snapshots/977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_pytorch_model.bin'
    )
    clip_model.dtype = torch.float32
    clip_model.visual.input_resolution = 224
    clip_model = clip_model.cuda()
    for params in clip_model.parameters():
        params.requires_grad_(False)

    model = NegaPromptCLIP(options, classnames, clip_model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    options.update(
        {
            'use_gpu': use_gpu
        }
    )
    """
    现在用新的
    """
    results = my_test_nega_clip(model, testloader)
    exit(0)


    print("Start Training!        ")
    training_main_loop(model, trainloader, testloader, run, options)


    exit(0)


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    img_size = 224
    results = dict()

    res = main_worker(options)
    sys.exit(0)


