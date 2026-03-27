#!/usr/bin/env python3
"""
Step 4: Generate FOA scenes using SpatialScaper.

For each scene in scene_plan.json:
1. Create SpatialScaper instance with the specified room
2. Add sources with controlled spatial positions
3. Generate FOA 4-channel audio (ACN/N3D: W, Y, Z, X)
4. Save labels in SELD format (100fps)
5. Save scene metadata as JSON

Output per scene:
    output/scenes/scene_XXXXXX/
        foa_WXYZ.wav       - 4-channel FOA audio at 48kHz
        labels.tsv         - SELD labels (frame, class_id, source_id, azimuth, elevation)
        meta.json          - Scene metadata
"""

import sys
import os
import json
import random
import traceback
from pathlib import Path
from collections import namedtuple
import multiprocessing as mp
from functools import partial

import numpy as np
import librosa
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "SpatialScaper-main"))

from config import (
    MEDLEY_PROCESSED_DIR,
    RIR_DIR,
    OUTPUT_DIR,
    SCENES_DIR,
    TARGET_SR,
    INSTRUMENT_MAP,
    LABEL_FPS,
)
from src.audio_utils import azel_to_room_xyz
from src.label_utils import save_seld_labels

from spatialscaper.core import Scaper, Event


def load_scene_plan(path=None):
    """Load scene plan from JSON."""
    if path is None:
        path = OUTPUT_DIR / "scene_plan.json"
    with open(path) as f:
        return json.load(f)


def get_room_bounds(room, rir_dir, fmt="foa", sr=TARGET_SR):
    """
    Get room XYZ bounds and receiver position by loading the room's RIR SOFA file.

    Returns:
        xyz_min, xyz_max, receiver_pos (center of bounds)
    """
    ssc = Scaper(
        duration=1.0,
        foreground_dir=str(MEDLEY_PROCESSED_DIR),
        rir_dir=str(rir_dir),
        fmt=fmt,
        room=room,
        sr=sr,
        DCASE_format=False,
        use_room_ambient_noise=False,
    )
    ssc.label_rate = LABEL_FPS
    xyz_min, xyz_max = ssc._get_room_min_max()
    receiver_pos = [(xyz_min[i] + xyz_max[i]) / 2 for i in range(3)]
    return xyz_min, xyz_max, receiver_pos


def generate_single_scene(scene, rir_dir, scenes_dir, foreground_dir, sr, label_fps):
    """
    Generate a single FOA scene.

    Args:
        scene: dict from scene_plan.json
        rir_dir: path to RIR datasets
        scenes_dir: output directory for scenes
        foreground_dir: path to processed audio files
        sr: sample rate
        label_fps: label frame rate

    Returns:
        (scene_id, success, message)
    """
    scene_name = scene["scene_name"]
    scene_dir = Path(scenes_dir) / scene_name
    audio_path = scene_dir / "foa_WXYZ.wav"
    label_path = scene_dir / "labels.tsv"
    meta_path = scene_dir / "meta.json"

    # Skip if already generated
    if audio_path.exists() and label_path.exists() and meta_path.exists():
        return (scene["scene_id"], True, "already exists")

    scene_dir.mkdir(parents=True, exist_ok=True)

    try:
        room = scene["room"]
        duration = scene["duration"]

        # Create SpatialScaper instance
        ssc = Scaper(
            duration=duration,
            foreground_dir=str(foreground_dir),
            rir_dir=str(rir_dir),
            fmt="foa",
            room=room,
            sr=sr,
            DCASE_format=False,
            max_event_overlap=3,
            max_event_dur=duration + 1.0,  # allow full duration events
            use_room_ambient_noise=False,
        )
        # SpatialScaper uses round(1/label_rate, 1) internally which fails for
        # rates > 10 (e.g., round(1/100, 1) = 0.0 → division by zero).
        # Use 10fps internally, then upsample labels to target rate.
        SS_LABEL_RATE = 10
        ssc.label_rate = SS_LABEL_RATE

        # Ensure fg_labels maps our instrument names to class IDs
        ssc.fg_labels = {name: idx for idx, name in INSTRUMENT_MAP.items()}

        # Get room spatial bounds
        xyz_min, xyz_max = ssc._get_room_min_max()
        receiver_pos = [(xyz_min[i] + xyz_max[i]) / 2 for i in range(3)]

        # Compute a safe distance that keeps sources within room bounds
        # Use half of the smallest room half-dimension
        room_half_dims = [(xyz_max[i] - xyz_min[i]) / 2 for i in range(3)]
        safe_distance = min(room_half_dims) * 0.8  # 80% of smallest half-dimension

        # Add sources directly as Event objects with controlled positions
        rng = random.Random(scene["scene_id"])

        for source_idx, source in enumerate(scene["sources"]):
            source_path = str(Path(foreground_dir) / source["source_file"])

            # Get clip duration
            clip_duration = librosa.get_duration(path=source_path)
            event_duration = min(clip_duration, duration)

            # Convert azimuth/elevation to room XYZ
            xyz = azel_to_room_xyz(
                source["azimuth_deg"],
                source["elevation_deg"],
                distance=safe_distance,
                receiver_pos=receiver_pos,
            )

            # Clamp to room bounds (with small margin)
            margin = 0.01
            xyz_clamped = [
                max(xyz_min[i] + margin, min(xyz_max[i] - margin, xyz[i]))
                for i in range(3)
            ]

            # Random SNR between 10-25 dB
            snr = rng.uniform(10, 25)

            ssc.fg_events.append(Event(
                label=source["instrument_name"],
                source_file=source_path,
                source_time=0,
                event_time=0.0,
                event_duration=event_duration,
                event_position=[xyz_clamped],  # list of [x,y,z] for static
                snr=snr,
                role="foreground",
                pitch_shift=None,
                time_stretch=None,
            ))

        # Generate FOA audio and labels
        # SpatialScaper's save_output appends .wav and .csv to paths
        ss_audio_stem = str(audio_path).replace(".wav", "")
        ss_label_stem = str(label_path).replace(".tsv", "")
        ssc.generate(ss_audio_stem, ss_label_stem)

        # Load SpatialScaper's raw labels
        # SS format: [frame, class_id, source_id, azimuth, elevation, distance]
        ss_label_csv = ss_label_stem + ".csv"
        raw_labels = np.loadtxt(ss_label_csv, delimiter=",")
        if raw_labels.ndim == 1:
            raw_labels = raw_labels.reshape(1, -1)

        # Extract SELD-relevant columns at SS_LABEL_RATE (10fps)
        # and upsample to target label_fps (100fps)
        upsample_factor = label_fps // SS_LABEL_RATE
        seld_labels = []
        for row in raw_labels:
            ss_frame = int(row[0])
            class_id = int(row[1])
            source_id = int(row[2])
            azimuth = round(float(row[3]), 1)
            elevation = round(float(row[4]), 1)

            base_frame = ss_frame * upsample_factor
            for sub in range(upsample_factor):
                seld_labels.append([
                    base_frame + sub,
                    class_id, source_id, azimuth, elevation,
                ])

        save_seld_labels(seld_labels, label_path)

        # Rename SS's output .wav (it appended .wav to our stem)
        ss_wav = ss_audio_stem + ".wav"
        if ss_wav != str(audio_path) and os.path.exists(ss_wav):
            os.rename(ss_wav, str(audio_path))

        # Clean up SS's CSV label file
        if os.path.exists(ss_label_csv):
            os.remove(ss_label_csv)

        # Save metadata
        meta = {
            "scene_id": scene["scene_id"],
            "scene_name": scene_name,
            "room": room,
            "hrtf_set": scene["hrtf_set"],
            "duration": duration,
            "sr": sr,
            "label_fps": label_fps,
            "num_sources": scene["num_sources"],
            "receiver_pos": receiver_pos,
            "room_bounds": {
                "min": xyz_min.tolist() if hasattr(xyz_min, 'tolist') else list(xyz_min),
                "max": xyz_max.tolist() if hasattr(xyz_max, 'tolist') else list(xyz_max),
            },
            "sources": [],
        }

        # Extract rendered DOA per source from labels
        # seld_labels rows: [frame, class_id, source_id, azimuth, elevation]
        rendered_doa = {}  # source_id -> (azimuth, elevation)
        for lbl in seld_labels:
            sid = int(lbl[2])
            if sid not in rendered_doa:
                rendered_doa[sid] = (lbl[3], lbl[4])

        for source_idx, source in enumerate(scene["sources"]):
            xyz = azel_to_room_xyz(
                source["azimuth_deg"],
                source["elevation_deg"],
                distance=safe_distance,
                receiver_pos=receiver_pos,
            )
            xyz_clamped = [
                max(xyz_min[i] + margin, min(xyz_max[i] - margin, xyz[i]))
                for i in range(3)
            ]
            # Use rendered DOA (consistent with labels.tsv) as primary
            r_az, r_el = rendered_doa.get(source_idx, (source["azimuth_deg"], source["elevation_deg"]))
            meta["sources"].append({
                "source_idx": source_idx,
                "instrument_class": source["instrument_class"],
                "instrument_name": source["instrument_name"],
                "azimuth_deg": r_az,
                "elevation_deg": r_el,
                "planned_azimuth_deg": source["azimuth_deg"],
                "planned_elevation_deg": source["elevation_deg"],
                "position_xyz": xyz_clamped,
                "source_file": source["source_file"],
            })

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return (scene["scene_id"], True, "ok")

    except Exception as e:
        tb = traceback.format_exc()
        return (scene["scene_id"], False, f"{e}\n{tb}")


def generate_scenes_worker(scene_batch, rir_dir, scenes_dir, foreground_dir, sr, label_fps):
    """Worker function for multiprocessing."""
    results = []
    for scene in scene_batch:
        result = generate_single_scene(scene, rir_dir, scenes_dir, foreground_dir, sr, label_fps)
        results.append(result)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate FOA scenes")
    parser.add_argument("--plan", default=None, help="Path to scene_plan.json")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--start", type=int, default=0, help="Start scene index")
    parser.add_argument("--end", type=int, default=None, help="End scene index (exclusive)")
    parser.add_argument("--test", action="store_true", help="Test mode: generate 3 scenes only")
    args = parser.parse_args()

    # Load scene plan
    scenes = load_scene_plan(args.plan)
    print(f"Loaded scene plan: {len(scenes)} scenes")

    # Subset
    if args.test:
        scenes = scenes[:3]
        print("TEST MODE: generating 3 scenes only")
    else:
        scenes = scenes[args.start:args.end]
        if args.end:
            print(f"Processing scenes {args.start} to {args.end}")

    print(f"\n{'=' * 60}")
    print(f"FOA Scene Generation")
    print(f"  Scenes to generate: {len(scenes)}")
    print(f"  Output: {SCENES_DIR}")
    print(f"  Workers: {args.workers}")
    print(f"  Sample rate: {TARGET_SR} Hz")
    print(f"  Label FPS: {LABEL_FPS}")
    print(f"{'=' * 60}\n")

    SCENES_DIR.mkdir(parents=True, exist_ok=True)

    if args.workers <= 1:
        # Sequential processing (easier to debug)
        results = []
        for scene in tqdm(scenes, desc="Generating scenes"):
            result = generate_single_scene(
                scene, str(RIR_DIR), str(SCENES_DIR),
                str(MEDLEY_PROCESSED_DIR), TARGET_SR, LABEL_FPS,
            )
            results.append(result)
            if not result[1]:
                print(f"\n  ERROR scene {result[0]}: {result[2][:200]}")
    else:
        # Parallel processing
        batch_size = max(1, len(scenes) // (args.workers * 4))
        batches = [scenes[i:i+batch_size] for i in range(0, len(scenes), batch_size)]

        worker_fn = partial(
            generate_scenes_worker,
            rir_dir=str(RIR_DIR),
            scenes_dir=str(SCENES_DIR),
            foreground_dir=str(MEDLEY_PROCESSED_DIR),
            sr=TARGET_SR,
            label_fps=LABEL_FPS,
        )

        results = []
        with mp.Pool(args.workers) as pool:
            for batch_results in tqdm(
                pool.imap_unordered(worker_fn, batches),
                total=len(batches),
                desc="Generating scenes",
            ):
                results.extend(batch_results)
                # Print errors immediately
                for r in batch_results:
                    if not r[1]:
                        print(f"\n  ERROR scene {r[0]}: {r[2][:200]}")

    # Summary
    success = sum(1 for r in results if r[1])
    skipped = sum(1 for r in results if r[1] and r[2] == "already exists")
    errors = sum(1 for r in results if not r[1])

    print(f"\n{'=' * 60}")
    print(f"Generation complete!")
    print(f"  Success: {success} ({success - skipped} new, {skipped} skipped)")
    print(f"  Errors: {errors}")
    if errors > 0:
        print(f"\n  First few errors:")
        for r in results:
            if not r[1]:
                print(f"    Scene {r[0]}: {r[2][:100]}")
                errors -= 1
                if errors <= 5:
                    break
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
