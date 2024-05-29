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

Hacked together by Ross Wightman (https://github.com/rwightman), and edited by
Edoardo Debenedetti (https://github.com/dedeswim) to work with Adversarial Training.

The original training script can be found at
https://github.com/rwightman/pytorch-image-models/, and the original license at
https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE.
"""
import logging
import shutil
from dataclasses import replace

import torch.nn as nn
from timm.bits import Monitor, TrainServices, distribute_bn, initialize_device
from timm.utils import random_seed, setup_default_logging, unwrap_model

from src import (  # noqa  # Models import needed to register the extra models that are not in timm
    attacks, models, utils)
from src.arg_parser import parse_args
from src.engine import evaluate, train_one_epoch
from src.setup_task import (resolve_attack_cfg, setup_checkpoints_output, setup_data, setup_train_task,
                            update_state_with_norm_model)

_logger = logging.getLogger('train')


def main():
    setup_default_logging()
    args, args_text = parse_args()

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp, channels_last=args.channels_last)
    if dev_env.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 device per process. '
                     'Process %d, total %d.' % (dev_env.global_rank, dev_env.world_size))
    else:
        _logger.info('Training with a single process on 1 device.')

    random_seed(args.seed, 0)  # Set all random seeds the same for model/state init (mandatory for XLA)

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    assert args.aug_splits == 0 or args.aug_splits > 1, 'A split of 1 makes no sense'

    train_state = setup_train_task(args, dev_env, mixup_active)
    train_cfg = train_state.train_cfg

    # Set random seeds across ranks differently for train
    # FIXME perhaps keep the same and just set diff seeds for dataloader worker process? what about TFDS?
    random_seed(args.seed, dev_env.global_rank)

    data_config, loader_eval, loader_train = setup_data(args,
                                                        unwrap_model(train_state.model).default_cfg, dev_env,
                                                        mixup_active)

    mean_feature_vectors = torch.zeros(size=(args.num_classes, args.num_classes), device=device)
    class_sample_counts = torch.zeros(size=(args.num_classes,), dtype=torch.long, device=device)

    for inputs, _ in loader_train:
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = train_state.model(inputs)
            # Accumulate logits and count samples for each class
            for class_idx in range(args.num_classes):
                class_feature_vectors = logits[(logits.argmax(dim=1) == class_idx)]
                mean_feature_vectors[class_idx] += torch.sum(class_feature_vectors, dim=0)
                class_sample_counts[class_idx] += class_feature_vectors.size(0)

    # Compute mean feature vectors
    for class_idx in range(args.num_classes):
        mean_feature_vectors[class_idx] /= class_sample_counts[class_idx]

    # # Multi sampled anchor inputs
    # # Initialize anchor_inputs and anchor_distances as lists of lists to store anchor inputs and distances for each class
    # anchor_inputs = [[] for _ in range(args.num_classes)]
    # anchor_distances = [[] for _ in range(args.num_classes)]

    # for inputs, targets in loader_train:
    #     inputs = inputs.to(device)
    #     with torch.no_grad():
    #         logits = train_state.model(inputs)
    #         # Compute anchor inputs and distances, and update them if necessary
    #         for class_idx in range(args.num_classes):
    #             class_logits = logits[(targets == class_idx)]
    #             if class_logits.size(0) > 0:  # Ensure there are samples for the class
    #                 # Compute distances from mean feature vector
    #                 distances = torch.norm(class_logits - mean_feature_vectors[class_idx], dim=1)
    #                 # Update anchor_inputs and anchor_distances if the new distances are smaller
    #                 for distance, input_tensor in zip(distances, inputs[(targets == class_idx)]):
    #                     if len(anchor_distances[class_idx]) < 3 or distance < max(anchor_distances[class_idx]):
    #                         anchor_inputs[class_idx].append(input_tensor.clone().detach())
    #                         anchor_distances[class_idx].append(distance.item())
    #                         if len(anchor_distances[class_idx]) > 3:
    #                             max_index = anchor_distances[class_idx].index(max(anchor_distances[class_idx]))
    #                             anchor_inputs[class_idx].pop(max_index)
    #                             anchor_distances[class_idx].pop(max_index)

    # # Convert anchor_inputs list of lists to a single tensor
    # anchor_inputs_tensor = torch.cat([torch.stack(anchor_inputs[i]) for i in range(args.num_classes)], dim=0)
    # train_state.anchor_inputs_tensor = anchor_inputs_tensor

    # # Single sampled anchor inputs
    # Initialize anchor_inputs as a list to store anchor inputs for each class 
    anchor_inputs = [None] * args.num_classes 

    for inputs, targets in loader_train: 
        inputs = inputs.to(device) 
        with torch.no_grad(): 
            logits = train_state.model(inputs) # Compute anchor inputs and store corresponding inputs 
            for class_idx in range(args.num_classes): 
                class_logits = logits[(targets == class_idx)] # Ensure there are multiple samples for the class 
                if class_logits.size(0) > 1: 
                    # Compute distances from mean feature vector 
                    distances = torch.norm(class_logits - mean_feature_vectors[class_idx], dim=1) 
                    closest_idx = distances.argmin() 
                    # Update anchor input if closer to mean feature vector 
                    if anchor_inputs[class_idx] is None or distances[closest_idx] < torch.norm(train_state.model(anchor_inputs[class_idx].unsqueeze(0)) - mean_feature_vectors[class_idx]): 
                        anchor_inputs[class_idx] = inputs[(targets == class_idx)][closest_idx].clone().detach() 

    # Convert anchor_inputs list to torch tensor 
    anchor_inputs_tensor = torch.stack(anchor_inputs) 
    train_state.anchor_inputs_tensor = anchor_inputs_tensor

    if args.normalize_model:
        train_state = update_state_with_norm_model(dev_env, train_state, data_config)

    # setup checkpoint manager
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    checkpoint_manager = None
    output_dir = None
    checkpoints_dir = None

    if dev_env.global_primary:
        checkpoint_manager, output_dir, checkpoints_dir = setup_checkpoints_output(
            vars(args), args_text, data_config, eval_metric)

    services = TrainServices(
        monitor=Monitor(output_dir=output_dir,
                        logger=_logger,
                        hparams=vars(args),
                        output_enabled=dev_env.primary,
                        experiment_name=args.experiment,
                        log_wandb=args.log_wandb and dev_env.global_primary),
        checkpoint=checkpoint_manager,  # type: ignore
    )

    if (wandb_run := services.monitor.wandb_run) is not None:
        assert output_dir is not None
        utils.write_wandb_info(args.run_notes, output_dir, wandb_run)

    if output_dir is not None and output_dir.startswith('gs://'):
        services.monitor.csv_writer = utils.GCSSummaryCsv(output_dir=output_dir)

    if args.adv_training is not None:
        eval_attack_cfg = resolve_attack_cfg(args, eval=True)
        attack_criterion = nn.NLLLoss(reduction="sum")
        dev_env.to_device(attack_criterion)
        eval_attack = attacks.make_attack(eval_attack_cfg.name,
                                          eval_attack_cfg.eps,
                                          eval_attack_cfg.step_size,
                                          eval_attack_cfg.steps,
                                          eval_attack_cfg.norm,
                                          eval_attack_cfg.boundaries,
                                          criterion=attack_criterion,
                                          dev_env=dev_env)
    else:
        eval_attack = None

    if dev_env.global_primary:
        _logger.info('Starting training, the first steps may take a long time')

    try:
        for epoch in range(train_state.epoch, train_cfg.num_epochs):
            if dev_env.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            if dev_env.distributed and isinstance(loader_train, utils.CombinedLoaders) and hasattr(
                    loader_train.sampler2, 'set_epoch'):
                loader_train.sampler2.set_epoch(epoch)

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
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(train_state.model, args.dist_bn == 'reduce', dev_env)

            eval_metrics = evaluate(train_state.model,
                                    train_state.eval_loss,
                                    loader_eval,
                                    train_state,
                                    services.monitor,
                                    dev_env,
                                    attack=eval_attack,
                                    log_interval=train_state.train_cfg.log_interval,
                                    use_mp_loader=args.use_mp_loader)

            if train_state.model_ema is not None:
                if dev_env.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(train_state.model_ema, args.dist_bn == 'reduce', dev_env)

                ema_eval_metrics = evaluate(train_state.model_ema.module,
                                            train_state.eval_loss,
                                            loader_eval,
                                            train_state,
                                            services.monitor,
                                            dev_env,
                                            phase_suffix='EMA',
                                            attack=eval_attack,
                                            log_interval=train_state.train_cfg.log_interval,
                                            use_mp_loader=args.use_mp_loader)
                eval_metrics = ema_eval_metrics

            if train_state.lr_scheduler is not None:
                # step LR for next epoch
                train_state.lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if services.monitor is not None:
                services.monitor.write_summary(index=epoch,
                                               results=dict(train=train_metrics, eval=eval_metrics))

            if checkpoint_manager is not None:
                # save proper checkpoint with eval metric
                best_checkpoint = checkpoint_manager.save_checkpoint(train_state, eval_metrics)
                best_metric, best_epoch = best_checkpoint.sort_key, best_checkpoint.epoch

            train_state = replace(train_state, epoch=epoch + 1)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if dev_env.global_primary and output_dir is not None and output_dir.startswith('gs://'):
        assert checkpoints_dir is not None
        try:
            _logger.info(f"Uploading checkpoints to {output_dir}")
            utils.upload_checkpoints_gcs(checkpoints_dir, output_dir)
            _logger.info(f"Uploaded checkpoints to {output_dir}, removing temporary dir")
            shutil.rmtree(checkpoints_dir)
        except Exception as e:
            _logger.exception(f"Failed to upload checkpoints to GCS: {e}. "
                              "Not removing the temporary dir {checkpoints_dir}.")

    if services.monitor.wandb_run is not None:
        services.monitor.wandb_run.finish()


def _mp_entry(*args):
    main()


if __name__ == '__main__':
    main()
