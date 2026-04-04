"""
Evaluate a CoupledFlow checkpoint on the Gray-Scott dataset.

Usage:
    python eval_coupled.py \
        --checkpoint ./output_grayscott/checkpoint-last.pth \
        --data_path ../data_generator/gray-scott/tiny_set/gs.pt \
        --split test \
        --batch_size 16
"""

import argparse
import logging
import sys

import torch
from data_loaders.grayscott_loader import build_grayscott_dataloader
from models.model_configs import instantiate_coupled_model


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint-last.pth")
    p.add_argument("--data_path", required=True, help="Path to gs.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--num_workers", default=2, type=int)
    p.add_argument("--device", default="cuda")
    return p


def rel_l2(pred, target):
    """Per-batch relative L2: accumulate numerator and denominator separately."""
    num = ((pred - target) ** 2).sum().item()
    den = (target ** 2).sum().item()
    return num, den


@torch.no_grad()
def evaluate(model, data_loader, device, net1, net2, label):
    model.eval()
    l2_num_1 = l2_den_1 = 0.0
    l2_num_2 = l2_den_2 = 0.0

    for batch in data_loader:
        source_1, source_2, target_1, target_2 = [x.to(device) for x in batch]
        pred_1, pred_2 = model.sample(source_1, source_2, net1=net1, net2=net2)

        n1, d1 = rel_l2(pred_1, target_1)
        n2, d2 = rel_l2(pred_2, target_2)
        l2_num_1 += n1; l2_den_1 += d1
        l2_num_2 += n2; l2_den_2 += d2

    rl2_1 = (l2_num_1 / max(l2_den_1, 1e-8)) ** 0.5
    rl2_2 = (l2_num_2 / max(l2_den_2, 1e-8)) ** 0.5
    rl2   = (rl2_1 + rl2_2) / 2

    logger.info(
        f"[{label:12s}]  rel-L2 = {rl2:.6f}  (u: {rl2_1:.6f}, v: {rl2_2:.6f})"
    )
    return rl2


def main():
    args = get_parser().parse_args()
    device = torch.device(args.device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt["args"]
    epoch = ckpt["epoch"]
    logger.info(f"Checkpoint from epoch {epoch}")

    logger.info(f"Building dataset split='{args.split}'")
    _, data_loader = build_grayscott_dataloader(
        data_path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    logger.info(f"Dataset size: {len(data_loader.dataset)}, batches: {len(data_loader)}")

    logger.info("Instantiating model")
    model = instantiate_coupled_model(saved_args).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    logger.info("Weights loaded")

    logger.info("=" * 60)
    nets_to_eval = [
        ("ema",    model.net1_ema, model.net2_ema),
        ("no-ema", model.net1,     model.net2),
    ]
    for i, decay in enumerate(model.ema_decays):
        nets_to_eval.append((
            f"ema{decay}",
            model._modules[f"net1_ema{i + 1}"],
            model._modules[f"net2_ema{i + 1}"],
        ))

    results = {}
    for label, net1, net2 in nets_to_eval:
        results[label] = evaluate(model, data_loader, device, net1, net2, label)

    logger.info("=" * 60)
    best = min(results, key=results.get)
    logger.info(f"Best: [{best}]  rel-L2 = {results[best]:.6f}")


if __name__ == "__main__":
    main()
