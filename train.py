#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
import yaml
from tensorflow.io import gfile
from timm.bits import (AccuracyTopK, AvgTensor, CheckpointManager, DeviceEnv,
                       Monitor, Tracker, TrainCfg, TrainServices, TrainState,
                       distribute_bn, initialize_device,
                       setup_model_and_optimizer)
from timm.data import (AugCfg, AugMixDataset, MixupCfg, create_loader_v2,
                       resolve_data_config)
from timm.data.dataset_factory import create_dataset
from timm.loss import *
from timm.models import convert_splitbn_model, create_model, safe_model_name
from timm.optim import optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import random_seed, setup_default_logging, unwrap_model
from torchvision import transforms

import attacks
import utils
from attacks import _SCHEDULES, AttackFn

_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config',
                                                 add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
parser.add_argument(
    '--dataset',
    '-d',
    metavar='NAME',
    default='',
    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split',
                    metavar='NAME',
                    default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split',
                    metavar='NAME',
                    default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model',
                    default='resnet50',
                    type=str,
                    metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument(
    '--pretrained',
    action='store_true',
    default=False,
    help='Start with pretrained version of specified network (if avail)')
parser.add_argument(
    '--initial-checkpoint',
    default='',
    type=str,
    metavar='PATH',
    help='Initialize model from this checkpoint (default: none)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='Resume full model and optimizer state from checkpoint (default: none)'
)
parser.add_argument(
    '--no-resume-opt',
    action='store_true',
    default=False,
    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes',
                    type=int,
                    default=None,
                    metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument(
    '--gp',
    default=None,
    type=str,
    metavar='POOL',
    help=
    'Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.'
)
parser.add_argument('--img-size',
                    type=int,
                    default=None,
                    metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument(
    '--input-size',
    default=None,
    nargs=3,
    type=int,
    metavar='N N N',
    help=
    'Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty'
)
parser.add_argument(
    '--crop-pct',
    default=None,
    type=float,
    metavar='N',
    help='Input image center crop percent (for validation only)')
parser.add_argument('--mean',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation',
                    default='',
                    type=str,
                    metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b',
                    '--batch-size',
                    type=int,
                    default=256,
                    metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb',
                    '--validation-batch-size',
                    type=int,
                    default=None,
                    metavar='N',
                    help='validation batch size override (default: None)')
parser.add_argument(
    '-nn',
    '--no-normalize',
    action='store_true',
    default=True,
    help='Avoids normalizing inputs (but it scales them in [0, 1]')

# Optimizer parameters
parser.add_argument('--opt',
                    default='sgd',
                    type=str,
                    metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps',
                    default=None,
                    type=float,
                    metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas',
                    default=None,
                    type=float,
                    nargs='+',
                    metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay',
                    type=float,
                    default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad',
                    type=float,
                    default=None,
                    metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument(
    '--clip-mode',
    type=str,
    default='norm',
    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched',
                    default='cosine',
                    type=str,
                    metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr',
                    type=float,
                    default=0.1,
                    metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise',
                    type=float,
                    nargs='+',
                    default=None,
                    metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct',
                    type=float,
                    default=0.67,
                    metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std',
                    type=float,
                    default=1.0,
                    metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul',
                    type=float,
                    default=1.0,
                    metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument(
    '--lr-cycle-decay',
    type=float,
    default=0.5,
    metavar='MULT',
    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit',
                    type=int,
                    default=1,
                    metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument(
    '--lr-k-decay',
    type=float,
    default=1.0,
    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr',
                    type=float,
                    default=0.0001,
                    metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument(
    '--min-lr',
    type=float,
    default=1e-5,
    metavar='LR',
    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs',
                    type=int,
                    default=300,
                    metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument(
    '--epoch-repeats',
    type=float,
    default=0.,
    metavar='N',
    help=
    'epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).'
)
parser.add_argument('--start-epoch',
                    default=None,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs',
                    type=float,
                    default=100,
                    metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs',
                    type=int,
                    default=5,
                    metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument(
    '--cooldown-epochs',
    type=int,
    default=10,
    metavar='N',
    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument(
    '--patience-epochs',
    type=int,
    default=10,
    metavar='N',
    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate',
                    '--dr',
                    type=float,
                    default=0.1,
                    metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument(
    '--no-aug',
    action='store_true',
    default=False,
    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale',
                    type=float,
                    nargs='+',
                    default=[0.08, 1.0],
                    metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio',
                    type=float,
                    nargs='+',
                    default=[3. / 4., 4. / 3.],
                    metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip',
                    type=float,
                    default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip',
                    type=float,
                    default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter',
                    type=float,
                    default=0.4,
                    metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument(
    '--aa',
    type=str,
    default=None,
    metavar='NAME',
    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument(
    '--aug-splits',
    type=int,
    default=0,
    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument(
    '--jsd-loss',
    action='store_true',
    default=False,
    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.'
)
parser.add_argument('--bce-loss',
                    action='store_true',
                    default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument(
    '--bce-target-thresh',
    type=float,
    default=None,
    help=
    'Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob',
                    type=float,
                    default=0.,
                    metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode',
                    type=str,
                    default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount',
                    type=int,
                    default=1,
                    help='Random erase count (default: 1)')
parser.add_argument(
    '--resplit',
    action='store_true',
    default=False,
    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup',
                    type=float,
                    default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix',
                    type=float,
                    default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument(
    '--cutmix-minmax',
    type=float,
    nargs='+',
    default=None,
    help=
    'cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
)
parser.add_argument(
    '--mixup-prob',
    type=float,
    default=1.0,
    help='Probability of performing mixup or cutmix when either/both is enabled'
)
parser.add_argument(
    '--mixup-switch-prob',
    type=float,
    default=0.5,
    help='Probability of switching to cutmix when both mixup and cutmix enabled'
)
parser.add_argument(
    '--mixup-mode',
    type=str,
    default='batch',
    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument(
    '--mixup-off-epoch',
    default=0,
    type=int,
    metavar='N',
    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing',
                    type=float,
                    default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument(
    '--train-interpolation',
    type=str,
    default='random',
    help='Training interpolation (random, bilinear, bicubic default: "random")'
)
parser.add_argument('--drop',
                    type=float,
                    default=0.0,
                    metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument(
    '--drop-connect',
    type=float,
    default=None,
    metavar='PCT',
    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path',
                    type=float,
                    default=None,
                    metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block',
                    type=float,
                    default=None,
                    metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument(
    '--bn-tf',
    action='store_true',
    default=False,
    help=
    'Use Tensorflow BatchNorm defaults for models that support it (default: False)'
)
parser.add_argument('--bn-momentum',
                    type=float,
                    default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps',
                    type=float,
                    default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn',
                    action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument(
    '--dist-bn',
    type=str,
    default='reduce',
    help=
    'Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")'
)
parser.add_argument('--split-bn',
                    action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema',
                    action='store_true',
                    default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument(
    '--model-ema-decay',
    type=float,
    default=0.9998,
    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    metavar='S',
                    help='random seed (default: 42)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=50,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--recovery-interval',
    type=int,
    default=0,
    metavar='N',
    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist',
                    type=int,
                    default=10,
                    metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument(
    '--save-images',
    action='store_true',
    default=False,
    help='save images of input bathes every log interval for debugging')
parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--channels-last',
                    action='store_true',
                    default=False,
                    help='Use channels_last memory layout')
parser.add_argument(
    '--pin-mem',
    action='store_true',
    default=False,
    help=
    'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
)
parser.add_argument('--output',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument(
    '--experiment',
    default='',
    type=str,
    metavar='NAME',
    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric',
                    default='top1',
                    type=str,
                    metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument(
    '--tta',
    type=int,
    default=0,
    metavar='N',
    help=
    'Test/inference time augmentation (oversampling) factor. 0=None (default: 0)'
)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    '--use-multi-epochs-loader',
    action='store_true',
    default=False,
    help=
    'use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript',
                    dest='torchscript',
                    action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--force-cpu',
                    action='store_true',
                    default=False,
                    help='Force CPU to be used even if HW accelerator exists.')
parser.add_argument('--log-wandb',
                    action='store_true',
                    default=False,
                    help='log training and validation metrics to wandb')
parser.add_argument('--run-notes',
                    default='',
                    type=str,
                    help='Description about this run')

# Adversarial training arguments
# Args for adversarial training:
parser.add_argument('--adv-training',
                    default=None,
                    type=str,
                    help='Enables adversarial training with the specified '
                    'technique (`trades` or `pgd`)')
parser.add_argument('--attack',
                    default='pgd',
                    type=str,
                    metavar='ATTACK',
                    help='What attack to use (default: "pgd")')
parser.add_argument('--attack-eps',
                    default=4 / 255,
                    type=float,
                    metavar='EPS',
                    help='The epsilon to use for the attack (default 8/255)')
parser.add_argument('--eps-schedule',
                    default='constant',
                    type=str,
                    metavar='SCHEDULE',
                    help='What schedule to use for eps (default: "constant")')
parser.add_argument('--eps-schedule-period',
                    default=10,
                    type=int,
                    metavar='EPOCHS',
                    help='How many epochs before reaching full eps')
parser.add_argument('--attack-lr',
                    default=1 / 255,
                    type=float,
                    metavar='ATTACK_LR',
                    help='Learning rate for the attack (default 1e-4)')
parser.add_argument('--attack-steps',
                    default=10,
                    type=int,
                    metavar='ATTACK_STEPS',
                    help='Number of steps to run attack for (default 10)')
parser.add_argument('--attack-norm',
                    default='linf',
                    type=str,
                    metavar='NORM',
                    help='The norm to use for the attack (default linf)')
parser.add_argument('--attack-boundaries',
                    default=(0, 1),
                    nargs=2,
                    type=int,
                    metavar='L H',
                    help='Boundaries of projection')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    dev_env = initialize_device(force_cpu=args.force_cpu,
                                amp=args.amp,
                                channels_last=args.channels_last)
    if dev_env.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process. Process %d, total %d.'
            % (dev_env.global_rank, dev_env.world_size))
    else:
        _logger.info('Training with a single process on 1 device.')

    random_seed(
        args.seed, 0
    )  # Set all random seeds the same for model/state init (mandatory for XLA)

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    assert args.aug_splits == 0 or args.aug_splits > 1, 'A split of 1 makes no sense'

    train_state = setup_train_task(args, dev_env, mixup_active)
    train_cfg = train_state.train_cfg

    # Set random seeds across ranks differently for train
    # FIXME perhaps keep the same and just set diff seeds for dataloader worker process? what about TFDS?
    random_seed(args.seed, dev_env.global_rank)

    data_config, loader_eval, loader_train = setup_data(
        args,
        unwrap_model(train_state.model).default_cfg, dev_env, mixup_active)

    # setup checkpoint manager
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    checkpoint_manager = None
    output_dir = None
    checkpoints_dir = None
    if dev_env.primary:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(
            args.output if args.output else './output/train',
            exp_name,
            inc=True)
        if output_dir.startswith("gs://"):
            checkpoints_dir = tempfile.mkdtemp()
            _logger.info(
                f"Temporarily saving checkpoints in {checkpoints_dir}")
        else:
            checkpoints_dir = output_dir
        checkpoint_manager = CheckpointManager(
            hparams=vars(args),
            checkpoint_dir=checkpoints_dir,
            recovery_dir=output_dir,
            metric_name=eval_metric,
            metric_decreasing=True if eval_metric == 'loss' else False,
            max_history=args.checkpoint_hist)

        if output_dir.startswith("gs://"):
            with gfile.GFile(os.path.join(output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)
        else:
            with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)

    services = TrainServices(
        monitor=Monitor(output_dir=output_dir,
                        logger=_logger,
                        hparams=vars(args),
                        output_enabled=dev_env.primary,
                        experiment_name=args.experiment,
                        log_wandb=args.log_wandb and dev_env.primary),
        checkpoint=checkpoint_manager,  # type: ignore
    )

    if (wandb_run := services.monitor.wandb_run) is not None:
        assert output_dir is not None
        # Log run notes and *true* output dir to wandb
        notes = args.run_notes
        if output_dir.startswith("gs://"):
            notes += f"\nBucket: {output_dir}\n"
        wandb_run.notes = args.run_notes  # type: ignore
        wandb_run.config.update({"output": output_dir}, allow_val_change=True)
        wandb_run_field = f"wandb_run: {wandb_run.url}\n"  # type: ignore
        # Log wandb run url to args file
        if output_dir.startswith("gs://"):
            with gfile.GFile(os.path.join(output_dir, 'args.yaml'), 'a') as f:
                f.write(wandb_run_field)
        else:
            with open(os.path.join(output_dir, 'args.yaml'), 'a') as f:
                f.write(wandb_run_field)

    if output_dir is not None and output_dir.startswith('gs://'):
        services.monitor.csv_writer = utils.GCSSummaryCsv(
            output_dir=output_dir)

    if args.adv_training is not None:
        attack_criterion = nn.NLLLoss(reduction="sum")
        dev_env.to_device(attack_criterion)
        eval_attack = attacks.make_attack(args.attack,
                                          args.attack_eps,
                                          args.attack_lr,
                                          args.attack_steps,
                                          args.attack_norm,
                                          args.attack_boundaries,
                                          criterion=attack_criterion)
    else:
        eval_attack = None

    _logger.info('Starting training, the first steps may take a long time')

    try:
        for epoch in range(train_state.epoch, train_cfg.num_epochs):
            if dev_env.distributed and hasattr(loader_train.sampler,
                                               'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
                if loader_train.mixup_enabled:
                    loader_train.mixup_enabled = False
            train_metrics = train_one_epoch(
                state=train_state,
                services=services,
                loader=loader_train,
                dev_env=dev_env,
            )

            if dev_env.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if dev_env.primary:
                    _logger.info(
                        "Distributing BatchNorm running means and vars")
                distribute_bn(train_state.model, args.dist_bn == 'reduce',
                              dev_env)

            eval_metrics = evaluate(train_state.model,
                                    train_state.eval_loss,
                                    loader_eval,
                                    services.monitor,
                                    dev_env,
                                    attack=eval_attack)

            if train_state.model_ema is not None:
                if dev_env.distributed and args.dist_bn in ('broadcast',
                                                            'reduce'):
                    distribute_bn(train_state.model_ema,
                                  args.dist_bn == 'reduce', dev_env)

                ema_eval_metrics = evaluate(train_state.model_ema.module,
                                            train_state.eval_loss,
                                            loader_eval,
                                            services.monitor,
                                            dev_env,
                                            phase_suffix='EMA',
                                            attack=eval_attack)
                eval_metrics = ema_eval_metrics

            if train_state.lr_scheduler is not None:
                # step LR for next epoch
                train_state.lr_scheduler.step(epoch + 1,
                                              eval_metrics[eval_metric])

            if services.monitor is not None:
                services.monitor.write_summary(index=epoch,
                                               results=dict(
                                                   train=train_metrics,
                                                   eval=eval_metrics))

            if checkpoint_manager is not None:
                # save proper checkpoint with eval metric
                best_checkpoint = checkpoint_manager.save_checkpoint(
                    train_state, eval_metrics)
                best_metric, best_epoch = best_checkpoint.sort_key, best_checkpoint.epoch

            train_state = replace(train_state, epoch=epoch + 1)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(
            best_metric, best_epoch))

    if dev_env.primary and output_dir is not None and output_dir.startswith(
            'gs://'):
        assert checkpoints_dir is not None
        try:
            _logger.info(f"Uploading checkpoints to {output_dir}")
            utils.upload_checkpoints_gcs(checkpoints_dir, output_dir)
            _logger.info(
                f"Uploaded checkpoints to {output_dir}, removing temporary dir"
            )
            shutil.rmtree(checkpoints_dir)
        except Exception as e:
            _logger.exception(
                f"Failed to upload checkpoints to GCS: {e}. "
                "Not removing the temporary dir {checkpoints_dir}.")

    if services.monitor.wandb_run is not None:
        services.monitor.wandb_run.finish()


def setup_train_task(args, dev_env: DeviceEnv, mixup_active: bool):

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if args.num_classes is None:
        assert hasattr(
            model, 'num_classes'
        ), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if dev_env.primary:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, '
            f'param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert args.aug_splits > 1
        model = convert_splitbn_model(model, max(args.aug_splits, 2))

    train_state = setup_model_and_optimizer(
        dev_env=dev_env,
        model=model,
        optimizer=args.opt,
        optimizer_cfg=optimizer_kwargs(cfg=args),
        clip_fn=args.clip_mode if args.clip_grad is not None else None,
        clip_value=args.clip_grad,
        model_ema=args.model_ema,
        model_ema_decay=args.model_ema_decay,
        resume_path=args.resume,
        use_syncbn=args.sync_bn,
        resume_opt=not args.no_resume_opt)

    # setup learning rate schedule and starting epoch
    # FIXME move into updater?
    lr_scheduler, num_epochs = create_scheduler(args,
                                                train_state.updater.optimizer)
    if lr_scheduler is not None and train_state.epoch > 0:
        lr_scheduler.step(train_state.epoch)

    # setup loss function
    if args.jsd_loss:
        assert args.aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=args.aug_splits,
                                        smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    eval_loss_fn = nn.CrossEntropyLoss()

    if args.adv_training is not None and args.adv_training == "pgd":
        attack_criterion = nn.NLLLoss(reduction="sum")
        train_attack = attacks.make_train_attack(args.attack,
                                                 args.eps_schedule,
                                                 args.attack_eps,
                                                 args.eps_schedule_period,
                                                 args.attack_lr,
                                                 args.attack_steps,
                                                 args.attack_norm,
                                                 args.attack_boundaries,
                                                 criterion=attack_criterion)
        compute_loss_fn = attacks.AdvTrainingLoss(train_attack, train_loss_fn)
    elif args.adv_training is not None and args.adv_training == "trades":
        attack_criterion = nn.KLDivLoss(reduction="sum")
        train_attack = attacks.make_train_attack(args.attack,
                                                 args.eps_schedule,
                                                 args.attack_eps,
                                                 args.eps_schedule_period,
                                                 args.attack_lr,
                                                 args.attack_steps,
                                                 args.attack_norm,
                                                 args.attack_boundaries,
                                                 criterion=attack_criterion)
        compute_loss_fn = attacks.TRADESLoss(train_attack, train_loss_fn, 6.0)
    else:
        compute_loss_fn = utils.ComputeLossFn(train_loss_fn)

    dev_env.to_device(train_loss_fn, eval_loss_fn, compute_loss_fn)

    if dev_env.primary:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    train_cfg = TrainCfg(
        num_epochs=num_epochs,
        log_interval=args.log_interval,
        recovery_interval=args.recovery_interval,
    )

    train_state = replace(
        train_state,
        lr_scheduler=lr_scheduler,
        train_loss=train_loss_fn,
        eval_loss=eval_loss_fn,
        train_cfg=train_cfg,
    )

    schedule = _SCHEDULES[args.eps_schedule](args.attack_eps,
                                             args.eps_schedule_period)

    train_state = utils.AdvTrainState.from_bits(
        train_state, compute_loss_fn=compute_loss_fn, eps_schedule=schedule)

    return train_state


def setup_data(args, default_cfg, dev_env: DeviceEnv, mixup_active: bool):
    data_config = resolve_data_config(vars(args),
                                      default_cfg=default_cfg,
                                      verbose=dev_env.primary)
    data_config['normalize'] = not args.no_normalize

    # create the train and eval datasets
    dataset_train = create_dataset(args.dataset,
                                   root=args.data_dir,
                                   split=args.train_split,
                                   is_training=True,
                                   batch_size=args.batch_size,
                                   repeats=args.epoch_repeats)

    dataset_eval = create_dataset(args.dataset,
                                  root=args.data_dir,
                                  split=args.val_split,
                                  is_training=False,
                                  batch_size=args.batch_size)

    # setup mixup / cutmix
    mixup_cfg = None
    if mixup_active:
        mixup_cfg = MixupCfg(prob=args.mixup_prob,
                             switch_prob=args.mixup_switch_prob,
                             mode=args.mixup_mode,
                             mixup_alpha=args.mixup,
                             cutmix_alpha=args.cutmix,
                             cutmix_minmax=args.cutmix_minmax,
                             label_smoothing=args.smoothing,
                             num_classes=args.num_classes)

    # wrap dataset in AugMix helper
    if args.aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train,
                                      num_splits=args.aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    if args.no_aug:
        train_aug_cfg = None
    else:
        train_aug_cfg = AugCfg(
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            ratio_range=args.ratio,
            scale_range=args.scale,
            hflip_prob=args.hflip,
            vflip_prob=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=args.aug_splits,
        )

    train_pp_cfg = utils.MyPreprocessCfg(
        input_size=data_config['input_size'],
        interpolation=train_interpolation,
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        aug=train_aug_cfg,
        normalize=data_config['normalize'],
    )

    # if using PyTorch XLA and RandomErasing is enabled, we must normalize and do RE in transforms on CPU
    normalize_in_transform = dev_env.type_xla and args.reprob > 0

    loader_train = create_loader_v2(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        normalize_in_transform=normalize_in_transform,
        pp_cfg=train_pp_cfg,
        mix_cfg=mixup_cfg,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader)

    if not train_pp_cfg.normalize:
        if normalize_in_transform:
            idx = -2 if args.reprob > 0 else -1
            loader_train.dataset.transform.transforms[
                idx] = transforms.ToTensor()
        else:
            loader_train.dataset.transform.transforms[
                -1] = transforms.ToTensor()
            loader_train.mean = None
            loader_train.std = None

    eval_pp_cfg = utils.MyPreprocessCfg(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        normalize=data_config['normalize'],
    )

    eval_workers = args.workers
    if 'tfds' in args.dataset:
        # FIXME reduce validation issues when using TFDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader_v2(
        dataset_eval,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        normalize_in_transform=normalize_in_transform,
        pp_cfg=eval_pp_cfg,
        num_workers=eval_workers,
        pin_memory=args.pin_mem,
    )

    if not eval_pp_cfg.normalize:
        loader_eval.dataset.transform.transforms[-1] = transforms.ToTensor()
        loader_eval.mean = None
        loader_eval.std = None

    return data_config, loader_eval, loader_train


def train_one_epoch(
    state: utils.AdvTrainState,
    services: TrainServices,
    loader,
    dev_env: DeviceEnv,
):
    tracker = Tracker()
    # FIXME move loss meter into task specific TaskMetric
    loss_meter = AvgTensor()
    accuracy_meter = AccuracyTopK()
    robust_accuracy_meter = AccuracyTopK()

    state.model.train()
    state.updater.reset()  # zero-grad

    step_end_idx = len(loader) - 1
    tracker.mark_iter()
    for step_idx, (sample, target) in enumerate(loader):
        tracker.mark_iter_data_end()

        # FIXME move forward + loss into model 'task' wrapper
        with dev_env.autocast():
            loss, output, adv_output = state.compute_loss_fn(
                state.model, sample, target, state.epoch)

        state.updater.apply(loss)

        tracker.mark_iter_step_end()

        state.updater.after_step(
            after_train_step,
            state,
            services,
            dev_env,
            step_idx,
            step_end_idx,
            tracker,
            loss_meter,
            accuracy_meter,
            robust_accuracy_meter,
            (output, adv_output, target, loss),
        )

        tracker.mark_iter()
        # end for

    if hasattr(state.updater.optimizer, 'sync_lookahead'):
        state.updater.optimizer.sync_lookahead()

    return OrderedDict([('loss', loss_meter.compute().item()),
                        ('eps', state.eps_schedule(state.epoch)),
                        ('lr', state.updater.get_average_lr())])


def after_train_step(
    state: TrainState,
    services: TrainServices,
    dev_env: DeviceEnv,
    step_idx: int,
    step_end_idx: int,
    tracker: Tracker,
    loss_meter: AvgTensor,
    accuracy_meter: AccuracyTopK,
    robust_accuracy_meter: AccuracyTopK,
    tensors: Tuple[torch.Tensor, ...],
):
    """
    After the core loss / backward / gradient apply step, we perform all non-gradient related
    activities here including updating meters, metrics, performing logging, and writing checkpoints.

    Many / most of these operations require tensors to be moved to CPU, they shoud not be done
    every step and for XLA use they should be done via the optimizer step_closure. This function includes
    everything that should be executed within the step closure.

    Args:
        state:
        services:
        dev_env:
        step_idx:
        step_end_idx:
        tracker:
        loss_meter:
        accuracy_meter:
        robust_accuracy_meter:
        tensors:

    Returns:

    """
    end_step = step_idx == step_end_idx

    with torch.no_grad():
        output, adv_output, target, loss = tensors
        loss_meter.update(loss, output.shape[0])

        if state.model_ema is not None:
            # FIXME should ema update be included here or in train / updater step? does it matter?
            state.model_ema.update(state.model)

        state = replace(state, step_count_global=state.step_count_global + 1)
        cfg = state.train_cfg

        if services.monitor is not None and end_step or (
                step_idx + 1) % cfg.log_interval == 0:
            global_batch_size = dev_env.world_size * output.shape[0]
            loss_avg = loss_meter.compute()

            if services.monitor is not None:
                lr_avg = state.updater.get_average_lr()
                services.monitor.log_step(
                    'Train',
                    step_idx=step_idx,
                    step_end_idx=step_end_idx,
                    epoch=state.epoch,
                    loss=loss_avg.item(),
                    rate=tracker.get_avg_iter_rate(global_batch_size),
                    lr=lr_avg)

        if services.checkpoint is not None and cfg.recovery_interval and (
                end_step or (step_idx + 1) % cfg.recovery_interval == 0):
            services.checkpoint.save_recovery(state)

        if state.lr_scheduler is not None:
            # FIXME perform scheduler update here or via updater after_step call?
            state.lr_scheduler.step_update(num_updates=state.step_count_global)


def evaluate(model: nn.Module,
             loss_fn: nn.Module,
             loader,
             logger: Monitor,
             dev_env: DeviceEnv,
             phase_suffix: str = '',
             log_interval: int = 10,
             attack: Optional[AttackFn] = None):
    tracker = Tracker()
    losses_m = AvgTensor()
    # FIXME move loss and accuracy modules into task specific TaskMetric obj
    accuracy_m = AccuracyTopK()
    robust_accuracy_m = AccuracyTopK()

    model.eval()

    end_idx = len(loader) - 1
    tracker.mark_iter()
    with torch.no_grad():
        for step_idx, (sample, target) in enumerate(loader):
            tracker.mark_iter_data_end()
            last_step = step_idx == end_idx

            with dev_env.autocast():
                output = model(sample)
                loss = loss_fn(output, target)

                if attack is not None:
                    with torch.enable_grad():
                        model.train()
                        adv_sample = attack(model, sample, target)
                        model.eval()
                        adv_output = model(adv_sample)
                else:
                    adv_output = None

            # FIXME, explictly marking step for XLA use since I'm not using the parallel xm loader
            # need to investigate whether parallel loader wrapper is helpful on tpu-vm or only use for 2-vm setup.
            if dev_env.type_xla:
                dev_env.mark_step()
            elif dev_env.type_cuda:
                dev_env.synchronize()

            # FIXME uncommenting this fixes race btw model `output`/`loss` and loss_m/accuracy_m meter input
            # for PyTorch XLA GPU use.
            # This issue does not exist for normal PyTorch w/ GPU (CUDA) or PyTorch XLA w/ TPU.
            # loss.item()

            tracker.mark_iter_step_end()
            losses_m.update(loss, output.size(0))
            accuracy_m.update(output, target)

            if adv_output is not None:
                robust_accuracy_m.update(adv_output, target)

            if last_step or step_idx % log_interval == 0:
                top1, top5 = accuracy_m.compute().values()
                if adv_output is not None:
                    robust_top1, _ = robust_accuracy_m.compute().values()
                else:
                    robust_top1 = None

                loss_avg = losses_m.compute()
                logger.log_step(
                    'Eval',
                    step_idx=step_idx,
                    step_end_idx=end_idx,
                    loss=loss_avg.item(),
                    top1=top1.item(),
                    top5=top5.item(),
                    robust_top1=robust_top1.item()
                    if robust_top1 is not None else None,
                    phase_suffix=phase_suffix,
                )
            tracker.mark_iter()

    top1, top5 = accuracy_m.compute().values()
    robust_top1, _ = robust_accuracy_m.compute().values()
    results = OrderedDict([
        ('loss', losses_m.compute().item()),
        ('top1', top1.item()),
        ('robust_top1', robust_top1.item()),
    ])
    return results


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()
