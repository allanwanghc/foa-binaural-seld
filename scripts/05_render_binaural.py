#!/usr/bin/env python3
"""
Step 5: Render binaural audio from FOA scenes.

For each scene:
1. Read meta.json to determine which HRTF set to use
2. Load foa_WXYZ.wav (4-channel FOA)
3. Apply SH-domain HRTF decoding → 2-channel binaural
4. Save binaural_LR.wav

Uses pre-initialized BinauralRenderer objects (one per HRTF set).
"""

import sys
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SCENES_DIR,
    HRTF_FILES,
    TARGET_SR,
)
from src.binaural_renderer import create_renderers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render binaural from FOA")
    parser.add_argument("--start", type=int, default=0, help="Start scene index")
    parser.add_argument("--end", type=int, default=None, help="End scene index")
    parser.add_argument("--test", action="store_true", help="Test mode: 3 scenes")
    args = parser.parse_args()

    # Find all scene directories
    scene_dirs = sorted(SCENES_DIR.glob("scene_*"))
    if args.test:
        scene_dirs = scene_dirs[:3]
    else:
        scene_dirs = scene_dirs[args.start:args.end]

    print(f"{'=' * 60}")
    print(f"Binaural Rendering")
    print(f"  Scenes to process: {len(scene_dirs)}")
    print(f"  HRTF sets: {list(HRTF_FILES.keys())}")
    print(f"{'=' * 60}\n")

    # Pre-initialize renderers
    print("Initializing HRTF renderers...")
    renderers = create_renderers(HRTF_FILES, TARGET_SR)
    print(f"\nRenderers ready: {list(renderers.keys())}\n")

    processed = 0
    skipped = 0
    errors = 0

    for scene_dir in tqdm(scene_dirs, desc="Rendering binaural"):
        binaural_path = scene_dir / "binaural_LR.wav"

        # Skip if already rendered
        if binaural_path.exists():
            skipped += 1
            continue

        meta_path = scene_dir / "meta.json"
        foa_path = scene_dir / "foa_WXYZ.wav"

        if not meta_path.exists() or not foa_path.exists():
            continue

        try:
            # Load metadata
            with open(meta_path) as f:
                meta = json.load(f)

            hrtf_set = meta["hrtf_set"]
            renderer = renderers[hrtf_set]

            # Load FOA audio
            foa_audio, sr = sf.read(str(foa_path))
            assert sr == TARGET_SR, f"Expected {TARGET_SR} Hz, got {sr}"
            assert foa_audio.ndim == 2 and foa_audio.shape[1] == 4, \
                f"Expected 4-channel FOA, got shape {foa_audio.shape}"

            # Render binaural
            binaural = renderer.render(foa_audio)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(binaural))
            if max_val > 0.99:
                binaural = binaural * 0.95 / max_val

            # Save
            sf.write(str(binaural_path), binaural, TARGET_SR, subtype="FLOAT")
            processed += 1

        except Exception as e:
            print(f"\n  ERROR {scene_dir.name}: {e}")
            errors += 1

    print(f"\n{'=' * 60}")
    print(f"Binaural rendering complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Errors: {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
