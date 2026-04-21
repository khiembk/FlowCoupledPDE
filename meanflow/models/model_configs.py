import copy
import torch.nn as nn
from models.meanflow import MeanFlow
from models.coupled_flow import CoupledFlow
from models.unet import SongUNet
from models.unet1d import SongUNet1d

MODEL_ARCHS = {
    "unet":       SongUNet,
    "unet32":     SongUNet,
    "unet32_lite": SongUNet,
    "unet_lite":  SongUNet,
    "unet1d":     SongUNet1d,
}

# Configs for single-process MeanFlow model
MODEL_CONFIGS = {
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
    "unet1d": {
        "seq_len":            256,
        "in_channels":        2,
        "out_channels":       1,
        "model_channels":     64,
        "channel_mult":       (1, 2, 2, 2),
        "num_blocks":         2,
        "attn_resolutions":   (32,),
        "channel_mult_noise": 2,
        "embedding_type":     "positional",
    },
}

# Configs for CoupledFlow (one entry per arch, shared by net1 and net2).
# dropout is injected at instantiation time from --dropout arg.
COUPLED_CONFIGS = {
    "unet": {
        "img_resolution": 64,       # GrayScott spatial resolution 64×64
        "in_channels": 2,
        "out_channels": 1,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
        "use_checkpoint": True,
    },
    "unet32": {
        "img_resolution": 32,       # tiny_set / small spatial resolution 32×32
        "in_channels": 2,
        "out_channels": 1,
        "channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
        "use_checkpoint": False,
    },
    # Lightweight 32×32 variant (~5M params/net) for tiny 2-process datasets.
    "unet32_lite": {
        "img_resolution": 32,
        "in_channels": 2,
        "out_channels": 1,
        "model_channels": 64,
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "channel_mult": [1, 2, 2],
        "num_blocks": 2,
        "encoder_type": "standard",
        "decoder_type": "standard",
        "use_checkpoint": False,
    },
    "unet1d": {
        "seq_len":            256,  # LV sequence length
        "in_channels":        2,
        "out_channels":       1,
        "model_channels":     64,
        "channel_mult":       (1, 2, 2, 2),
        "num_blocks":         2,
        "attn_resolutions":   (32,),
        "channel_mult_noise": 2,
        "embedding_type":     "positional",
    },
    # Lightweight 2D UNet for large N-process systems (e.g., THM 5-process).
    # model_channels=64, channel_mult=[1,2,2], num_blocks=2 → ~6M params per net
    # vs ~50M for "unet"; 5 such nets ≈ 30M trainable params total.
    # in_channels is overridden to n_proc at instantiation time.
    "unet_lite": {
        "img_resolution": 64,
        "in_channels": 2,
        "out_channels": 1,
        "model_channels": 64,
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "channel_mult": [1, 2, 2],
        "num_blocks": 2,
        "encoder_type": "standard",
        "decoder_type": "standard",
        "use_checkpoint": True,
    },
    # BZ: 3-process 1D system — used by CoupledFlowBZ (separate from CoupledFlow)
    "unet1d_bz": {
        "seq_len":            256,
        "in_channels":        3,    # 3 BZ processes concatenated
        "out_channels":       1,
        "model_channels":     64,
        "channel_mult":       (1, 2, 2, 2),
        "num_blocks":         2,
        "attn_resolutions":   (32,),
        "channel_mult_noise": 2,
        "embedding_type":     "positional",
    },
}


def instantiate_model(args) -> nn.Module:
    architechture = args.arch
    assert architechture in MODEL_CONFIGS, \
        f"Model architecture {architechture} is missing its config."

    configs = copy.deepcopy(MODEL_CONFIGS[architechture])
    configs['dropout'] = args.dropout
    arch = MODEL_ARCHS[architechture]
    if args.use_edm_aug:
        configs['augment_dim'] = 6
    return MeanFlow(arch=arch, net_configs=configs, args=args)


def instantiate_coupled_model(args) -> nn.Module:
    architechture = args.arch
    assert architechture in COUPLED_CONFIGS, \
        f"Architecture '{architechture}' has no coupled config. " \
        f"Available: {list(COUPLED_CONFIGS.keys())}"

    arch = MODEL_ARCHS[architechture]
    n1_configs = copy.deepcopy(COUPLED_CONFIGS[architechture])
    n2_configs = copy.deepcopy(COUPLED_CONFIGS[architechture])
    n1_configs['dropout'] = args.dropout
    n2_configs['dropout'] = args.dropout
    if args.use_edm_aug:
        n1_configs['augment_dim'] = 6
        n2_configs['augment_dim'] = 6

    return CoupledFlow(arch=arch, net1_configs=n1_configs, net2_configs=n2_configs, args=args)


def instantiate_coupled_nproc_model(args, n_proc: int) -> nn.Module:
    """
    Instantiate CoupledFlowNProc for n_proc processes using the unet_lite arch.
    Each net receives all n_proc channels concatenated (in_channels=n_proc).
    """
    from models.coupled_flow_nproc import CoupledFlowNProc

    cfg = copy.deepcopy(COUPLED_CONFIGS["unet_lite"])
    cfg["in_channels"] = n_proc
    cfg["dropout"] = args.dropout

    net_configs_list = [copy.deepcopy(cfg) for _ in range(n_proc)]
    return CoupledFlowNProc(arch=SongUNet, args=args, net_configs_list=net_configs_list)


def instantiate_coupled_bz_model(args) -> nn.Module:
    """Instantiate CoupledFlowBZ (3-process) for BZ training. Separate from instantiate_coupled_model."""
    from models.coupled_flow_bz import CoupledFlowBZ
    cfg = copy.deepcopy(COUPLED_CONFIGS["unet1d_bz"])
    n1 = copy.deepcopy(cfg); n1['dropout'] = args.dropout
    n2 = copy.deepcopy(cfg); n2['dropout'] = args.dropout
    n3 = copy.deepcopy(cfg); n3['dropout'] = args.dropout
    return CoupledFlowBZ(arch=SongUNet1d, net1_configs=n1, net2_configs=n2, net3_configs=n3, args=args)
