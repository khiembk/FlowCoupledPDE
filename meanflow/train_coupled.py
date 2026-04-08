import datetime
import logging
import os
import sys
import time
from pathlib import Path

from functools import partial
from data_loaders.grayscott_loader import GrayScottCoupledDataset, build_grayscott_dataloader
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_coupled_model
from train_arg_parser import get_args_parser
from training import distributed_mode
from training.data_transform import get_transform_cifar, get_transform_mnist

from training.load_and_save import load_model, save_model
from training.coupled_training_loop import train_coupled_one_epoch, train_combined_loss_step, train_local_loss_step, evaluate_coupled_rel_l2
from torchmetrics.aggregation import MeanMetric
import models.rng as rng

from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision('high')

logger = logging.getLogger(__name__)


def print_model(model):
    logger.info("=" * 91)
    num_params = 0
    for name, param in model.named_parameters():
        param_std = param.std().item()
        if param.requires_grad:
            num_params += param.numel()
            logger.info(f"{name:48} | {str(list(param.shape)):24} | std: {param_std:.6f}")
    logger.info("=" * 91)
    logger.info(f"Total params: {num_params}")


def get_data_loader(args, is_for_fid):
    
    if args.dataset == "grayscott":
        print("load grayscott...")
        train_set, train_loader = build_grayscott_dataloader(
            data_path=args.data_path,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            horizon=1,
            normalize=False,
            train_ratio=getattr(args, "train_ratio", 0.8),
            val_ratio=getattr(args, "val_ratio", 0.1),
        )
        return train_loader



    if args.dataset == "cifar10":
        transforms = get_transform_cifar(is_for_fid)
        dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    elif args.dataset == "mnist":  # 3x32x32 MNIST for fast development
        transforms = get_transform_mnist()
        dataset = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    logger.info(dataset)

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        worker_init_fn=partial(rng.worker_init_fn, rank=global_rank),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=not is_for_fid,  # for FID evaluation, we want to keep all samples
    )
    logger.info(str(sampler))
    return data_loader


def main(args):
    print("init distributed mode")
    distributed_mode.init_distributed_mode(args)

    print(f"Rank: {distributed_mode.get_rank()}")
    print(f"World Size: {distributed_mode.get_world_size()}")

    if distributed_mode.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logger.addHandler(logging.NullHandler())

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        # create tensorboard
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
        logger.info(f"Tensorboard writer created at {args.output_dir}")
    else:
        log_writer = None
        logger.info('Writer not created.')

    device = torch.device(args.device)

    # set the seeds
    seed = args.seed + distributed_mode.get_rank()  # legacy. TODO: rng.fold_in 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    data_loader_train = get_data_loader(args, is_for_fid=False)
    data_loader_fid = get_data_loader(args, is_for_fid=True)

    if args.dataset == "grayscott":
        logger.info("Building Gray-Scott test dataloader for evaluation")
        _, data_loader_val = build_grayscott_dataloader(
            data_path=args.data_path,
            split="test",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            horizon=1,
            normalize=False,
            shuffle=False,
            drop_last=False,
            train_ratio=getattr(args, "train_ratio", 0.8),
            val_ratio=getattr(args, "val_ratio", 0.1),
        )
        logger.info(f"Test dataset size: {len(data_loader_val.dataset)}, batches: {len(data_loader_val)}")
    else:
        data_loader_val = None

    # define the model
    logger.info("Initializing Model")
    model = instantiate_coupled_model(args)

    model.to(device)

    model_without_ddp = model
    print_model(model)

    eff_batch_size = args.batch_size * distributed_mode.get_world_size()

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
            gradient_as_bucket_view=True
        )
        model_without_ddp = model.module

    opt_params = list(model_without_ddp.net1.parameters()) + list(model_without_ddp.net2.parameters())
    if getattr(args, "use_gp", False):
        opt_params += list(model_without_ddp.gp_12.parameters()) + list(model_without_ddp.gp_21.parameters())
    optimizer = torch.optim.Adam(
        opt_params,
        lr=args.lr,
        betas=args.optimizer_betas,
        weight_decay=0.0,
    )

    warmup_iters = args.warmup_epochs * len(data_loader_train)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8 / args.lr, end_factor=1.0, total_iters=warmup_iters,)
    main_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=args.epochs * len(data_loader_train), factor=1.0)
    lr_schedule = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    if getattr(args, "auto_resume", False) and not args.resume:
        last_ckpt = os.path.join(args.output_dir, "checkpoint-last.pth")
        if os.path.isfile(last_ckpt):
            args.resume = last_ckpt
            logger.info(f"Auto-resuming from {last_ckpt}")
        else:
            logger.info("Auto-resume enabled but no checkpoint-last.pth found, starting from scratch.")

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
    )

    compiled_train_step = torch.compile(
        train_combined_loss_step,
        disable=not args.compile,
    )

    batch_loss = MeanMetric().to(device, non_blocking=True)
    batch_time = MeanMetric().to(device, non_blocking=True)
    batch_loss.reset()
    batch_time.reset()

    meters = {'batch_loss': batch_loss, 'batch_time': batch_time,}

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_coupled_one_epoch(
                model=model,
                compiled_train_step=compiled_train_step,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                log_writer=log_writer,
                args=args,
                meters=meters
            )

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    epoch=epoch,
                )
                logging.info(f"Saved checkpoint to {args.output_dir}")

            # Eval coupled model (rel-L2 on val split for grayscott, train for others):
            eval_loader = data_loader_val if data_loader_val is not None else data_loader_fid
            eval_split = "val" if data_loader_val is not None else "train"

            eval_nets = [
                ("ema", model_without_ddp.net1_ema, model_without_ddp.net2_ema),
                ("noema", model_without_ddp.net1, model_without_ddp.net2),
            ]
            for i, decay in enumerate(model_without_ddp.ema_decays):
                eval_nets.append((
                    f"ema{decay}",
                    model_without_ddp._modules[f"net1_ema{i + 1}"],
                    model_without_ddp._modules[f"net2_ema{i + 1}"],
                ))

            for suffix, net1_eval, net2_eval in eval_nets:
                rel_l2, rel_l2_1, rel_l2_2 = evaluate_coupled_rel_l2(
                    model_without_ddp, eval_loader, device, net1_eval, net2_eval, args
                )
                logging.info(
                    f"Eval epoch {epoch + 1} [{suffix}] ({eval_split}): "
                    f"rel-L2 = {rel_l2:.6f} (u: {rel_l2_1:.6f}, v: {rel_l2_2:.6f})"
                )
                if log_writer is not None:
                    log_writer.add_scalar(f"rel_L2_{suffix}", rel_l2, epoch + 1)
                    log_writer.add_scalar(f"rel_L2_u_{suffix}", rel_l2_1, epoch + 1)
                    log_writer.add_scalar(f"rel_L2_v_{suffix}", rel_l2_2, epoch + 1)

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
