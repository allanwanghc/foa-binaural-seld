#!/usr/bin/env python3
"""
Step 7: Finalize dataset splits (train/val/test).

Reads the pre-split mapping from 03c_assign_clips.py, verifies that
all scenes were generated successfully, and produces the final splits.json.

If pre_splits.json does not exist, falls back to stratified splitting
by room × hrtf_set (but clips will NOT be leak-proof in that case).
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OUTPUT_DIR,
    SCENES_DIR,
    SPLIT_RATIOS,
    SPLIT_SEED,
)


def main():
    print(f"{'=' * 60}")
    print(f"Dataset Split (finalize)")
    print(f"  Ratios: {SPLIT_RATIOS}")
    print(f"  Seed: {SPLIT_SEED}")
    print(f"{'=' * 60}\n")

    # Load scene plan
    plan_path = OUTPUT_DIR / "scene_plan.json"
    with open(plan_path) as f:
        scenes = json.load(f)
    scene_lookup = {s["scene_name"]: s for s in scenes}

    # Only include scenes that were actually generated
    valid_names = set()
    for scene in scenes:
        scene_dir = SCENES_DIR / scene["scene_name"]
        if (scene_dir / "foa_WXYZ.wav").exists():
            valid_names.add(scene["scene_name"])

    print(f"Valid scenes (with generated audio): {len(valid_names)}")

    # Try to load pre-splits from 03c_assign_clips.py
    pre_splits_path = OUTPUT_DIR / "pre_splits.json"
    if pre_splits_path.exists():
        print(f"Using pre-split mapping from: {pre_splits_path}")
        with open(pre_splits_path) as f:
            pre_splits = json.load(f)

        # Filter to only valid (generated) scenes
        splits = {}
        for split_name, scene_names in pre_splits.items():
            splits[split_name] = sorted(
                [n for n in scene_names if n in valid_names]
            )
    else:
        print("WARNING: pre_splits.json not found. Falling back to "
              "stratified split (clips may leak across splits).")

        valid_scenes = [s for s in scenes if s["scene_name"] in valid_names]
        strata = defaultdict(list)
        for scene in valid_scenes:
            key = f"{scene['room']}_{scene['hrtf_set']}"
            strata[key].append(scene["scene_name"])

        rng = np.random.RandomState(SPLIT_SEED)
        splits = {"train": [], "val": [], "test": []}

        for stratum_key, scene_names in sorted(strata.items()):
            rng.shuffle(scene_names)
            n = len(scene_names)
            n_train = int(n * SPLIT_RATIOS["train"])
            n_val = int(n * SPLIT_RATIOS["val"])

            splits["train"].extend(scene_names[:n_train])
            splits["val"].extend(scene_names[n_train:n_train + n_val])
            splits["test"].extend(scene_names[n_train + n_val:])

        for key in splits:
            splits[key].sort()

    # Save
    splits_path = OUTPUT_DIR / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    total = sum(len(v) for v in splits.values())
    print(f"\nSplit results:")
    for split_name, scene_list in splits.items():
        pct = 100 * len(scene_list) / total if total > 0 else 0
        print(f"  {split_name}: {len(scene_list)} scenes ({pct:.1f}%)")

    # Stratification check
    print(f"\nStratification check (room × hrtf per split):")
    for split_name in ["train", "val", "test"]:
        combo_counts = defaultdict(int)
        for scene_name in splits[split_name]:
            s = scene_lookup.get(scene_name, {})
            combo_counts[f"{s.get('room', '?')}_{s.get('hrtf_set', '?')}"] += 1
        print(f"  {split_name}:")
        for combo, count in sorted(combo_counts.items()):
            print(f"    {combo}: {count}")

    print(f"\nSaved to: {splits_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
