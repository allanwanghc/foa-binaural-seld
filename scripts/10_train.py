"""
Training script for SELD models with Multi-ACCDOA ADPIT loss.

Usage:
    python scripts/10_train.py --model seldnet --format foa --exp_name seldnet_foa
    python scripts/10_train.py --model conformer --format binaural --exp_name conformer_bin
    python scripts/10_train.py --model cst_former --format foa --exp_name cst_foa

Training config:
    - Optimizer: AdamW, lr=1e-3, weight_decay=1e-2
    - Scheduler: CosineAnnealingWarmRestarts (T_0=10)
    - Batch size: 32
    - Epochs: 100
    - Early stopping: patience=15 on SELD_score
    - Gradient clipping: max_norm=5.0
    - TensorBoard logging
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.seld_dataset import SELDDataset
from src.data.data_utils import collate_fn, compute_norm_stats
from src.models.seldnet import SELDNet
from src.models.conformer_seld import ConformerSELD
from src.models.cst_former import CSTFormer
from src.losses.seld_loss import ADPITLoss
from src.metrics.seld_metrics import SELDMetrics, multi_accdoa_to_events, label_to_events


def get_model(model_name, in_channels, num_classes=8):
    """Create model based on name."""
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


def get_scene_lists(data_dir, fmt):
    """
    Get train/val/test scene lists from data directory structure.
    Expects data_dir/features/{fmt}/ and data_dir/labels/

    Returns train_scenes, val_scenes, test_scenes as lists of filenames.
    """
    feature_dir = os.path.join(data_dir, 'features', fmt)
    label_dir = os.path.join(data_dir, 'labels')

    # Try to load split files if they exist
    split_dir = os.path.join(data_dir, 'splits')
    if os.path.exists(split_dir):
        train_scenes = _load_split(os.path.join(split_dir, 'train.txt'))
        val_scenes = _load_split(os.path.join(split_dir, 'val.txt'))
        test_scenes = _load_split(os.path.join(split_dir, 'test.txt'))
        return train_scenes, val_scenes, test_scenes

    # Otherwise, list all .npy files and split
    all_scenes = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
    # Filter to scenes that have both features and labels
    all_scenes = [s for s in all_scenes if os.path.exists(os.path.join(feature_dir, s))]

    n = len(all_scenes)
    if n == 0:
        raise RuntimeError(f"No matching .npy files found in {feature_dir} and {label_dir}")

    # Default split: 70% train, 15% val, 15% test
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_scenes = all_scenes[:n_train]
    val_scenes = all_scenes[n_train:n_train + n_val]
    test_scenes = all_scenes[n_train + n_val:]

    if not val_scenes:
        val_scenes = train_scenes[-1:]
    if not test_scenes:
        test_scenes = val_scenes

    return train_scenes, val_scenes, test_scenes


def _load_split(path):
    """Load scene list from text file."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_model(model, dataloader, criterion, device, num_classes):
    """
    Evaluate model on a dataset split.

    Returns:
        avg_loss: average loss over the dataset
        metrics: dict with ER, F, LE, LR, SELD_score
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    seld_metrics = SELDMetrics(nb_classes=num_classes)

    with torch.no_grad():
        for feat, label in dataloader:
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

    return avg_loss, metrics


def train(args):
    """Main training function."""
    # Setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'processed')
    exp_dir = os.path.join(project_root, 'experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)

    # Feature format
    fmt = args.format  # 'foa' or 'binaural'
    in_channels = 7 if fmt == 'foa' else 6
    feature_dir = os.path.join(data_dir, 'features', fmt)
    label_dir = os.path.join(data_dir, 'labels')

    # Allow overriding data directory
    if args.data_dir:
        data_dir = args.data_dir
        feature_dir = os.path.join(data_dir, 'features', fmt)
        label_dir = os.path.join(data_dir, 'labels')

    print(f"Feature dir: {feature_dir}")
    print(f"Label dir: {label_dir}")

    # Scene lists
    train_scenes, val_scenes, test_scenes = get_scene_lists(
        data_dir if args.data_dir else os.path.join(project_root, 'data', 'processed'),
        fmt
    )
    print(f"Train: {len(train_scenes)} scenes, Val: {len(val_scenes)} scenes, "
          f"Test: {len(test_scenes)} scenes")

    # Compute normalization stats on training data
    print("Computing normalization statistics...")
    norm_stats = compute_norm_stats(feature_dir, train_scenes)
    # Save norm stats for evaluation
    np.save(os.path.join(exp_dir, 'norm_mean.npy'), norm_stats[0])
    np.save(os.path.join(exp_dir, 'norm_std.npy'), norm_stats[1])

    # Datasets
    train_dataset = SELDDataset(
        feature_dir=feature_dir,
        label_dir=label_dir,
        scene_list=train_scenes,
        chunk_length=args.chunk_length,
        hop_length=args.chunk_length // 2,  # 50% overlap
        norm_stats=norm_stats,
        mode='train'
    )

    val_dataset = SELDDataset(
        feature_dir=feature_dir,
        label_dir=label_dir,
        scene_list=val_scenes,
        chunk_length=args.chunk_length,
        hop_length=args.chunk_length,  # No overlap for val
        norm_stats=norm_stats,
        mode='eval'
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda
    )

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")

    # Model
    num_classes = args.num_classes
    model = get_model(args.model, in_channels, num_classes)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, Parameters: {n_params:,}")

    # Loss
    criterion = ADPITLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1
    )

    # TensorBoard
    writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))

    # Save config
    config = vars(args)
    config['in_channels'] = in_channels
    config['n_params'] = n_params
    config['device'] = str(device)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_seld_score = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping patience: {args.patience}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]",
                     leave=False)
        for feat, label in pbar:
            feat = feat.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(feat)
            loss = criterion(output, label)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # Validate
        val_loss, val_metrics = evaluate_model(
            model, val_loader, criterion, device, num_classes
        )

        epoch_time = time.time() - epoch_start

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/ER', val_metrics['ER'], epoch)
        writer.add_scalar('Metrics/F1', val_metrics['F'], epoch)
        writer.add_scalar('Metrics/LE', val_metrics['LE'], epoch)
        writer.add_scalar('Metrics/LR', val_metrics['LR'], epoch)
        writer.add_scalar('Metrics/SELD_score', val_metrics['SELD_score'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"SELD: {val_metrics['SELD_score']:.4f} | "
              f"ER: {val_metrics['ER']:.2f} | "
              f"F1: {val_metrics['F']:.2f} | "
              f"LE: {val_metrics['LE']:.1f} | "
              f"LR: {val_metrics['LR']:.2f} | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping check
        if val_metrics['SELD_score'] < best_seld_score:
            best_seld_score = val_metrics['SELD_score']
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_seld_score': best_seld_score,
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, os.path.join(exp_dir, 'checkpoints', 'best_model.pth'))
            print(f"  >> New best SELD score: {best_seld_score:.4f} - Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_seld_score': best_seld_score,
                'config': config
            }
            torch.save(checkpoint,
                        os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch{epoch}.pth'))

    writer.close()

    print("\n" + "=" * 80)
    print(f"Training complete. Best SELD score: {best_seld_score:.4f}")
    print(f"Best model saved to: {os.path.join(exp_dir, 'checkpoints', 'best_model.pth')}")
    print(f"TensorBoard logs: {os.path.join(exp_dir, 'tensorboard')}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description='SELD Training Script')

    # Model
    parser.add_argument('--model', type=str, default='seldnet',
                        choices=['seldnet', 'conformer', 'cst_former'],
                        help='Model architecture')
    parser.add_argument('--format', type=str, default='foa',
                        choices=['foa', 'binaural'],
                        help='Audio format (foa or binaural)')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name for output directory')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of sound event classes')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory path')
    parser.add_argument('--chunk_length', type=int, default=500,
                        help='Chunk length in frames (500 = 5s at 100fps)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for AdamW')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
