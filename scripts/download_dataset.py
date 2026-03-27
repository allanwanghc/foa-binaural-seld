#!/usr/bin/env python3
"""
Download dataset from HuggingFace to local directory.

Usage:
    python scripts/download_dataset.py --repo awhc0813/foa-binaural-seld-dataset --output output/
    python scripts/download_dataset.py --repo awhc0813/foa-binaural-seld-dataset --output output/ --token YOUR_HF_TOKEN
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Download SELD dataset from HuggingFace"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="awhc0813/foa-binaural-seld-dataset",
        help="HuggingFace repository ID (default: awhc0813/foa-binaural-seld-dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Local directory to download into (default: output/)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (optional, for private repos)",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Repository type (default: dataset)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from: {args.repo}")
    print(f"Downloading to:   {output_dir}")
    print(f"Repo type:        {args.repo_type}")
    if args.token:
        print("Using provided HuggingFace token")
    print()

    try:
        local_dir = snapshot_download(
            repo_id=args.repo,
            repo_type=args.repo_type,
            local_dir=str(output_dir),
            token=args.token,
        )
        print(f"\nDownload complete. Files saved to: {local_dir}")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
