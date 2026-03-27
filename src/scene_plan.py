"""
Scene parameter pre-sampling for 10,000 scenes.

Generates a deterministic scene plan ensuring balanced coverage of
experimental factors: room × hrtf_set × instrument classes × spatial positions.

Each scene specifies: room, HRTF set, duration, number of sources,
and per-source instrument class, azimuth, elevation, and source audio file.
"""

import json
import random
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ROOMS,
    HRTF_SETS,
    INSTRUMENT_MAP,
    INSTRUMENT_NAMES,
    NUM_CLASSES,
    AZIMUTHS,
    ELEVATIONS,
    NUM_SOURCES_RANGE,
    DURATION_RANGE,
    NUM_SCENES,
    SPLIT_SEED,
    MEDLEY_PROCESSED_DIR,
    OUTPUT_DIR,
)


def collect_source_files():
    """
    Collect all available processed audio files organized by instrument.
    Returns dict: {instrument_name: [relative_path, ...]}
    """
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


def generate_scene_plan(num_scenes=NUM_SCENES, seed=SPLIT_SEED):
    """
    Generate a scene plan with balanced experimental factors.

    Strategy:
    - Cycle through room × hrtf_set combinations (3×3=9) to ensure balance
    - For each scene, sample 2-3 sources with distinct spatial positions
    - Instruments are sampled to maximize diversity across scenes
    - Source files are drawn without replacement per instrument (reset when exhausted)
    - Duration uniformly sampled from DURATION_RANGE

    Returns:
        list of scene dicts
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Collect available source files (for stats only; clip assignment
    # happens later in 07_split_dataset.py to prevent clip-level leakage)
    source_files = collect_source_files()
    print("Source files per instrument:")
    for name, files in source_files.items():
        print(f"  {name}: {len(files)}")

    # Room × HRTF combinations for balanced cycling
    room_hrtf_combos = list(itertools.product(ROOMS, HRTF_SETS))
    # 9 combinations → cycle through them
    combo_cycle = itertools.cycle(room_hrtf_combos)

    # Track instrument usage for balancing
    instrument_usage = defaultdict(int)

    scenes = []

    for scene_idx in range(num_scenes):
        # Balanced room × hrtf assignment
        room, hrtf_set = next(combo_cycle)

        # Random duration
        duration = round(rng.uniform(*DURATION_RANGE), 2)

        # Number of sources: 2 or 3
        num_sources = rng.randint(*NUM_SOURCES_RANGE)

        # Select instruments for this scene (no duplicates within a scene)
        # Prefer under-represented instruments
        usage_counts = [instrument_usage[name] for name in INSTRUMENT_NAMES]
        min_usage = min(usage_counts)
        # Instruments at or near minimum usage get priority
        weights = [1.0 / (1 + count - min_usage) for count in usage_counts]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        chosen_instruments = list(np_rng.choice(
            INSTRUMENT_NAMES,
            size=num_sources,
            replace=False,
            p=probs,
        ))

        # Generate unique spatial positions for each source
        available_positions = list(itertools.product(AZIMUTHS, ELEVATIONS))
        rng.shuffle(available_positions)
        chosen_positions = available_positions[:num_sources]

        # Build source list
        sources = []
        for i, inst_name in enumerate(chosen_instruments):
            class_id = INSTRUMENT_NAMES.index(inst_name)
            az, el = chosen_positions[i]

            sources.append({
                "instrument_class": class_id,
                "instrument_name": inst_name,
                "azimuth_deg": az,
                "elevation_deg": el,
                "source_file": None,  # assigned in 07_split_dataset.py
            })

            instrument_usage[inst_name] += 1

        scene = {
            "scene_id": scene_idx,
            "scene_name": f"scene_{scene_idx:06d}",
            "room": room,
            "hrtf_set": hrtf_set,
            "duration": duration,
            "num_sources": num_sources,
            "sources": sources,
        }
        scenes.append(scene)

    return scenes


def print_plan_stats(scenes):
    """Print statistics about the generated scene plan."""
    print(f"\n{'=' * 60}")
    print(f"Scene Plan Statistics")
    print(f"{'=' * 60}")
    print(f"Total scenes: {len(scenes)}")

    # Room distribution
    room_counts = defaultdict(int)
    for s in scenes:
        room_counts[s["room"]] += 1
    print(f"\nRoom distribution:")
    for room, count in sorted(room_counts.items()):
        print(f"  {room}: {count} ({100*count/len(scenes):.1f}%)")

    # HRTF distribution
    hrtf_counts = defaultdict(int)
    for s in scenes:
        hrtf_counts[s["hrtf_set"]] += 1
    print(f"\nHRTF distribution:")
    for hrtf, count in sorted(hrtf_counts.items()):
        print(f"  {hrtf}: {count} ({100*count/len(scenes):.1f}%)")

    # Room × HRTF
    combo_counts = defaultdict(int)
    for s in scenes:
        combo_counts[(s["room"], s["hrtf_set"])] += 1
    print(f"\nRoom × HRTF distribution:")
    for (room, hrtf), count in sorted(combo_counts.items()):
        print(f"  {room} × {hrtf}: {count}")

    # Sources per scene
    src_counts = defaultdict(int)
    for s in scenes:
        src_counts[s["num_sources"]] += 1
    print(f"\nSources per scene:")
    for n, count in sorted(src_counts.items()):
        print(f"  {n} sources: {count} ({100*count/len(scenes):.1f}%)")

    # Instrument distribution (across all sources)
    inst_counts = defaultdict(int)
    for s in scenes:
        for src in s["sources"]:
            inst_counts[src["instrument_name"]] += 1
    total_sources = sum(inst_counts.values())
    print(f"\nInstrument distribution (total source appearances: {total_sources}):")
    for name, count in sorted(inst_counts.items()):
        print(f"  {name}: {count} ({100*count/total_sources:.1f}%)")

    # Duration stats
    durations = [s["duration"] for s in scenes]
    print(f"\nDuration: min={min(durations):.2f}s, max={max(durations):.2f}s, "
          f"mean={np.mean(durations):.2f}s")

    # Azimuth distribution
    all_az = [src["azimuth_deg"] for s in scenes for src in s["sources"]]
    print(f"\nAzimuth range: [{min(all_az)}, {max(all_az)}] deg")
    print(f"Elevation range: [{min(src['elevation_deg'] for s in scenes for src in s['sources'])}, "
          f"{max(src['elevation_deg'] for s in scenes for src in s['sources'])}] deg")

    print(f"{'=' * 60}")


def save_scene_plan(scenes, output_path=None):
    """Save scene plan to JSON."""
    if output_path is None:
        output_path = OUTPUT_DIR / "scene_plan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"\nScene plan saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    scenes = generate_scene_plan()
    print_plan_stats(scenes)
    save_scene_plan(scenes)
