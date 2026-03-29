"""
Evaluation script for SELD models.

Loads the best checkpoint from an experiment and evaluates on a specified split.
Computes and prints all SELD metrics (ER, F1, LE, LR, SELD_score).

Usage:
    python scripts/11_evaluate.py --exp_name seldnet_foa --split test
    python scripts/11_evaluate.py --exp_name conformer_bin --split val
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.seld_dataset import SELDDataset
from src.data.data_utils import collate_fn
from src.models.seldnet import SELDNet
from src.models.conformer_seld import ConformerSELD
from src.models.cst_former import CSTFormer
from src.losses.seld_loss import ADPITLoss
from src.metrics.seld_metrics import SELDMetrics, multi_accdoa_to_events, label_to_events


def get_model(model_name, in_channels, num_classes=8):
    """Create model based on name (must match training config)."""
    if model_name == 'seldnet':
        return SELDNet(
            in_channels=in_channels,
            num_classes=num_classes,
            num_tracks=3,
            cnn_filters=64,
            rnn_size=128,
            rnn_layers=2,
            attn_layers=2,
            attn_heads=8,
            fnn_size=128,
            dropout=0.05,
            f_pool_sizes=[4, 4, 2],
            t_pool_sizes=[1, 1, 1],
            n_freq_bins=64
        )
    elif model_name == 'conformer':
        return ConformerSELD(
            in_channels=in_channels,
            num_classes=num_classes,
            num_tracks=3,
            cnn_filters=128,
            d_model=256,
            n_heads=8,
            n_conformer_layers=4,
            conv_kernel_size=31,
            d_ff=1024,
            dropout=0.1,
            f_pool_sizes=[4, 4],
            n_freq_bins=64
        )
    elif model_name == 'cst_former':
        return CSTFormer(
            in_channels=in_channels,
            num_classes=num_classes,
            num_tracks=3,
            cnn_filters=64,
            n_attn_layers=2,
            n_heads=8,
            fnn_size=256,
            dropout=0.1,
            f_pool_sizes=[4, 4, 2],
            t_pool_sizes=[1, 1, 1],
            n_freq_bins=64,
            chunk_length=500,
            patch_size_t=25
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_scene_list_for_split(data_dir, fmt, split):
    """Get scene list for a specific split."""
    feature_dir = os.path.join(data_dir, fmt)
    label_dir = os.path.join(data_dir, 'labels')

    # Try splits.json first (our pipeline output)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_path = os.path.join(project_root, 'output', 'splits.json')
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        scenes = [f"{sid}.npy" for sid in splits.get(split, [])
                  if os.path.exists(os.path.join(feature_dir, f"{sid}.npy"))
                  and os.path.exists(os.path.join(label_dir, f"{sid}.npy"))]
        return scenes

    # Fallback: list all and split
    all_scenes = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
    all_scenes = [s for s in all_scenes if os.path.exists(os.path.join(feature_dir, s))]

    n = len(all_scenes)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    if split == 'train':
        return all_scenes[:n_train]
    elif split == 'val':
        return all_scenes[n_train:n_train + n_val]
    elif split == 'test':
        return all_scenes[n_train + n_val:]
    else:
        raise ValueError(f"Unknown split: {split}")


def evaluate(args):
    """Main evaluation function."""
    # Setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = os.path.join(project_root, 'output', 'experiments', args.exp_name)

    if not os.path.exists(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Load config
    config_path = os.path.join(exp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config.get('model', args.model)
        fmt = config.get('format', args.format)
        num_classes = config.get('num_classes', 8)
        in_channels = config.get('in_channels', 7 if fmt == 'foa' else 6)
        chunk_length = config.get('chunk_length', 500)
    else:
        model_name = args.model
        fmt = args.format
        num_classes = 8
        in_channels = 7 if fmt == 'foa' else 6
        chunk_length = 500

    print(f"Model: {model_name}, Format: {fmt}, Classes: {num_classes}")

    # Data directory
    data_dir = args.data_dir if args.data_dir else os.path.join(project_root, 'output', 'features')
    feature_dir = os.path.join(data_dir, fmt)
    label_dir = os.path.join(data_dir, 'labels')

    # Load normalization stats
    norm_mean_path = os.path.join(exp_dir, 'norm_mean.npy')
    norm_std_path = os.path.join(exp_dir, 'norm_std.npy')
    if os.path.exists(norm_mean_path) and os.path.exists(norm_std_path):
        norm_stats = (np.load(norm_mean_path), np.load(norm_std_path))
        print("Loaded normalization stats from experiment directory")
    else:
        norm_stats = None
        print("WARNING: No normalization stats found, running without normalization")

    # Scene list
    scenes = get_scene_list_for_split(data_dir, fmt, args.split)
    print(f"Evaluating on {args.split} split: {len(scenes)} scenes")

    # Dataset and DataLoader
    dataset = SELDDataset(
        feature_dir=feature_dir,
        label_dir=label_dir,
        scene_list=scenes,
        chunk_length=chunk_length,
        hop_length=chunk_length,  # No overlap for evaluation
        norm_stats=norm_stats,
        mode='eval'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda
    )

    print(f"Evaluation chunks: {len(dataset)}")

    # Model
    model = get_model(model_name, in_channels, num_classes)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    if args.checkpoint:
        checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_seld_score' in checkpoint:
        print(f"Checkpoint best SELD score (val): {checkpoint['best_seld_score']:.4f}")

    # Loss function
    criterion = ADPITLoss()

    # Evaluate
    model.eval()
    total_loss = 0
    n_batches = 0
    seld_metrics = SELDMetrics(nb_classes=num_classes)

    print("\nRunning evaluation...")
    with torch.no_grad():
        for feat, label in tqdm(dataloader, desc="Evaluating"):
            feat = feat.to(device)
            label = label.to(device)

            output = model(feat)
            loss = criterion(output, label)
            total_loss += loss.item()
            n_batches += 1

            # Convert to events for metric computation
            output_np = output.detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()

            for b in range(output_np.shape[0]):
                pred_events = multi_accdoa_to_events(output_np[b], num_classes)
                gt_events = label_to_events(label_np[b], num_classes)
                seld_metrics.update(pred_events, gt_events)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = seld_metrics.compute()

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results on {args.split} split")
    print(f"Experiment: {args.exp_name}")
    print(f"Model: {model_name}")
    print("=" * 60)
    print(f"  Loss:       {avg_loss:.4f}")
    print(f"  SELD Score: {metrics['SELD_score']:.4f}")
    print(f"  Error Rate: {metrics['ER']:.4f}")
    print(f"  F-score:    {metrics['F']:.4f} ({100 * metrics['F']:.1f}%)")
    print(f"  LE (deg):   {metrics['LE']:.1f}")
    print(f"  LR:         {metrics['LR']:.4f} ({100 * metrics['LR']:.1f}%)")
    print("=" * 60)

    # Save results
    results_dir = os.path.join(exp_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'split': args.split,
        'loss': avg_loss,
        'metrics': metrics,
        'checkpoint': checkpoint_path,
        'n_scenes': len(scenes),
        'n_chunks': len(dataset)
    }

    results_path = os.path.join(results_dir, f'{args.split}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='SELD Evaluation Script')

    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate on')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint (default: best_model.pth)')

    # Optional overrides (auto-loaded from config if available)
    parser.add_argument('--model', type=str, default='seldnet',
                        choices=['seldnet', 'conformer', 'cst_former'],
                        help='Model architecture (overrides config)')
    parser.add_argument('--format', type=str, default='foa',
                        choices=['foa', 'binaural'],
                        help='Audio format (overrides config)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory path')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
