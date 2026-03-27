#!/usr/bin/env python3
"""
Step 3c: Split clips into train/val/test pools and assign to scenes.

Must run AFTER scene plan generation (scene_plan.py) and BEFORE scene
generation (04_generate_scenes.py).

Two-phase approach to prevent clip-level data leakage:
  Phase 1 — Determine which scenes go to train/val/test (by room × hrtf_set).
  Phase 2 — Split clips per instrument into disjoint pools,
            then assign clips only from the matching pool.

This guarantees that no audio clip appears in more than one split.
"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    OUTPUT_DIR,
    SPLIT_RATIOS,
    SPLIT_SEED,
    INSTRUMENT_NAMES,
    MEDLEY_PROCESSED_DIR,
)


# ─── Clip collection ────────────────────────────────────────────────

def collect_source_files():
    """Collect all processed audio files organized by instrument."""
    source_files = {}
    for inst_name in INSTRUMENT_NAMES:
        inst_dir = MEDLEY_PROCESSED_DIR / inst_name
        if inst_dir.exists():
            files = sorted([
                f"{inst_name}/{f.name}"
                for f in inst_dir.glob("*.wav")
            ])
            source_files[inst_name] = files
        else:
            source_files[inst_name] = []
    return source_files


def split_clip_pools(source_files, ratios, seed):
    """
    Split clips per instrument into train/val/test pools (no overlap).
    Returns: {"train": {inst: [clip, ...]}, "val": {...}, "test": {...}}
    """
    rng = np.random.RandomState(seed)
    pools = {"train": {}, "val": {}, "test": {}}

    for inst_name, files in source_files.items():
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        pools["train"][inst_name] = files[:n_train]
        pools["val"][inst_name] = files[n_train:n_train + n_val]
        pools["test"][inst_name] = files[n_train + n_val:]

    return pools


def assign_clips_to_scenes(scenes, split_name, clip_pools, seed):
    """
    Assign clips from the correct pool to scenes in this split.
    Uses no-replacement draw with reshuffle on exhaustion.
    """
    rng = random.Random(seed)

    # Build per-instrument draw pools
    draw_pools = {}
    for inst, clips in clip_pools[split_name].items():
        pool = list(clips)
        rng.shuffle(pool)
        draw_pools[inst] = pool

    for scene in scenes:
        for src in scene["sources"]:
            inst = src["instrument_name"]
            if len(draw_pools[inst]) == 0:
                # Reshuffle the full pool for this split
                draw_pools[inst] = list(clip_pools[split_name][inst])
                rng.shuffle(draw_pools[inst])
            src["source_file"] = draw_pools[inst].pop()


# ─── Verification ────────────────────────────────────────────────────

def verify_no_clip_leakage(scenes_by_split):
    """Assert that no clip appears in more than one split."""
    clips_by_split = {}
    for split_name, scene_list in scenes_by_split.items():
        clips = set()
        for scene in scene_list:
            for src in scene["sources"]:
                clips.add(src["source_file"])
        clips_by_split[split_name] = clips

    splits = list(clips_by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = clips_by_split[splits[i]] & clips_by_split[splits[j]]
            if overlap:
                raise AssertionError(
                    f"Clip leakage between {splits[i]} and {splits[j]}: "
                    f"{len(overlap)} shared clips! Examples: {list(overlap)[:5]}"
                )
    print("  ✓ Zero clip leakage across splits")

    for split_name, clips in clips_by_split.items():
        print(f"    {split_name}: {len(clips)} unique clips used")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    print(f"{'=' * 60}")
    print(f"Clip Assignment (leak-proof)")
    print(f"  Split ratios: {SPLIT_RATIOS}")
    print(f"  Seed: {SPLIT_SEED}")
    print(f"{'=' * 60}\n")

    # Load scene plan (source_file should be null)
    plan_path = OUTPUT_DIR / "scene_plan.json"
    with open(plan_path) as f:
        scenes = json.load(f)

    print(f"Total scenes in plan: {len(scenes)}")

    # Check that clips haven't been assigned yet
    unassigned = sum(
        1 for s in scenes for src in s["sources"]
        if src.get("source_file") is None
    )
    total_sources = sum(len(s["sources"]) for s in scenes)
    print(f"Unassigned sources: {unassigned}/{total_sources}")
    if unassigned == 0:
        print("WARNING: All sources already have clips assigned. "
              "Re-assigning with leak-proof pools.")

    # ── Phase 1: Determine scene splits by room × hrtf_set ──

    strata = defaultdict(list)
    for scene in scenes:
        key = f"{scene['room']}_{scene['hrtf_set']}"
        strata[key].append(scene)

    rng = np.random.RandomState(SPLIT_SEED)
    scenes_by_split = {"train": [], "val": [], "test": []}

    for stratum_key, stratum_scenes in sorted(strata.items()):
        rng.shuffle(stratum_scenes)
        n = len(stratum_scenes)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        scenes_by_split["train"].extend(stratum_scenes[:n_train])
        scenes_by_split["val"].extend(stratum_scenes[n_train:n_train + n_val])
        scenes_by_split["test"].extend(stratum_scenes[n_train + n_val:])

    print(f"\nPhase 1 — Scene split:")
    for split_name, scene_list in scenes_by_split.items():
        print(f"  {split_name}: {len(scene_list)} scenes "
              f"({100 * len(scene_list) / len(scenes):.1f}%)")

    # ── Phase 2: Split clips, then assign to scenes ──

    source_files = collect_source_files()
    clip_pools = split_clip_pools(source_files, SPLIT_RATIOS, SPLIT_SEED)

    print(f"\nPhase 2 — Clip pools (per instrument):")
    for inst in INSTRUMENT_NAMES:
        t = len(clip_pools["train"].get(inst, []))
        v = len(clip_pools["val"].get(inst, []))
        te = len(clip_pools["test"].get(inst, []))
        print(f"  {inst}: train={t}, val={v}, test={te}")

    # Assign clips from matching pool (different seed offset per split)
    for i, split_name in enumerate(["train", "val", "test"]):
        assign_clips_to_scenes(
            scenes_by_split[split_name],
            split_name,
            clip_pools,
            seed=SPLIT_SEED + i + 100,
        )

    # ── Verify no leakage ──

    print(f"\nClip leakage check:")
    verify_no_clip_leakage(scenes_by_split)

    # ── Save pre-split mapping (for 07_split_dataset.py to use later) ──

    pre_splits = {}
    for split_name, scene_list in scenes_by_split.items():
        pre_splits[split_name] = sorted([s["scene_name"] for s in scene_list])

    pre_splits_path = OUTPUT_DIR / "pre_splits.json"
    with open(pre_splits_path, "w") as f:
        json.dump(pre_splits, f, indent=2)

    # ── Update scene_plan.json with assigned clips ──

    with open(plan_path, "w") as f:
        json.dump(scenes, f, indent=2)

    print(f"\nSaved pre-split mapping to: {pre_splits_path}")
    print(f"Updated scene plan with clip assignments: {plan_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
