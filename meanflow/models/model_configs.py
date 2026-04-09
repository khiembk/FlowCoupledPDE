# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from models.meanflow import MeanFlow
from models.coupled_flow import CoupledFlow
from models.unet import SongUNet
from models.unet1d import SongUNet1d

MODEL_ARCHS = {
    "unet":   SongUNet,
    "unet1d": SongUNet1d,
}

MODEL_CONFIGS = {
    "unet1d": {
        "seq_len":            256,
        "in_channels":        2,
        "out_channels":       1,
        "model_channels":     64,
        "channel_mult":       (1, 2, 2, 2),
        "num_blocks":         2,
        "attn_resolutions":   (32,),
        "dropout":            0.1,
        "channel_mult_noise": 2,
        "embedding_type":     "positional",
    },
    "unet": {
        "img_resolution": 32,
        "in_channels": 3,
        "out_channels": 3,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
    },
}

net1_configs = {
    "img_resolution": 64,   # GrayScott spatial resolution is 64×64
    "in_channels": 2,
    "out_channels": 1,
    "channel_mult_noise": 2,
    "resample_filter": [1, 3, 3, 1],
    "channel_mult": [2, 2, 2],
    "encoder_type": "standard",
    "decoder_type": "standard",
    "dropout": 0.2,
    "use_checkpoint": True, # gradient checkpointing: trade ~30% compute for ~5x activation memory reduction
}

net2_configs = {
    "img_resolution": 64,   # GrayScott spatial resolution is 64×64
    "in_channels": 2,
    "out_channels": 1,
    "channel_mult_noise": 2,
    "resample_filter": [1, 3, 3, 1],
    "channel_mult": [2, 2, 2],
    "encoder_type": "standard",
    "decoder_type": "standard",
    "dropout": 0.2,
    "use_checkpoint": True, # gradient checkpointing: trade ~30% compute for ~5x activation memory reduction
}



def instantiate_model(args) -> nn.Module:
    architechture = args.arch
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    configs = MODEL_CONFIGS[architechture]
    configs['dropout'] = args.dropout
    arch = MODEL_ARCHS[architechture]
    if args.use_edm_aug:
        configs['augment_dim'] = 6
    model = MeanFlow(arch=arch, net_configs=configs, args=args)

    return model


def instantiate_coupled_model(args) -> nn.Module:
    architechture = args.arch
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."

    arch = MODEL_ARCHS[architechture]

    if architechture == "unet1d":
        import copy
        cfg = copy.deepcopy(MODEL_CONFIGS["unet1d"])
        cfg["dropout"] = args.dropout
        n1_configs = cfg
        n2_configs = copy.deepcopy(cfg)
    else:
        if args.use_edm_aug:
            net1_configs['augment_dim'] = 6
            net2_configs['augment_dim'] = 6
        n1_configs = net1_configs
        n2_configs = net2_configs

    model = CoupledFlow(arch=arch, net1_configs=n1_configs, net2_configs=n2_configs, args=args)
    return model
