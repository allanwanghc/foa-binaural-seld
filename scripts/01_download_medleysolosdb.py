#!/usr/bin/env python3
"""
Step 1: Download Medley-solos-DB, filter 8 instrument classes,
        loudness-normalize to -23 LUFS, resample to 48 kHz,
        and organize by instrument subdirectory.

Output structure:
    data/medley_solos_db/processed/{instrument_name}/*.wav
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MEDLEY_RAW_DIR,
    MEDLEY_PROCESSED_DIR,
    TARGET_SR,
    TARGET_LUFS,
    INSTRUMENT_MAP,
)
from src.audio_utils import load_audio, normalize_loudness, resample_audio, save_audio


# Medley-solos-DB instrument ID to name mapping
# (matches the dataset's own encoding)
MEDLEY_INSTRUMENT_IDS = {
    0: "clarinet",
    1: "distorted_electric_guitar",
    2: "female_singer",
    3: "flute",
    5: "piano",
    6: "tenor_saxophone",
    7: "trumpet",
    8: "violin",
}


def download_medley_solos_db():
    """Download Medley-solos-DB using mirdata."""
    import mirdata

    print("=" * 60)
    print("Downloading Medley-solos-DB...")
    print(f"  Target directory: {MEDLEY_RAW_DIR}")
    print("=" * 60)

    MEDLEY_RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = mirdata.initialize(
        "medley_solos_db",
        data_home=str(MEDLEY_RAW_DIR),
    )
    dataset.download()

    print("Download complete.")
    return dataset


def process_clips(dataset):
    """
    Process all clips: loudness normalize and resample.
    Save to processed directory organized by instrument class.
    """
    print("=" * 60)
    print("Processing clips: normalize + resample...")
    print(f"  Target LUFS: {TARGET_LUFS}")
    print(f"  Target SR: {TARGET_SR} Hz")
    print(f"  Output: {MEDLEY_PROCESSED_DIR}")
    print("=" * 60)

    # Create output subdirectories for each instrument
    for inst_name in INSTRUMENT_MAP.values():
        (MEDLEY_PROCESSED_DIR / inst_name).mkdir(parents=True, exist_ok=True)

    # Get all track IDs
    track_ids = dataset.track_ids
    tracks = dataset.load_tracks()

    processed_count = 0
    skipped_count = 0

    for track_id in tqdm(track_ids, desc="Processing clips"):
        track = tracks[track_id]

        # Get instrument ID from the track
        instrument_id = track.instrument_id

        # Check if this instrument is in our target set
        if instrument_id not in MEDLEY_INSTRUMENT_IDS:
            skipped_count += 1
            continue

        instrument_name = MEDLEY_INSTRUMENT_IDS[instrument_id]

        # Get the audio file path
        audio_path = track.audio_path
        if audio_path is None or not os.path.exists(audio_path):
            skipped_count += 1
            continue

        # Load audio
        try:
            audio, orig_sr = load_audio(audio_path)
        except Exception as e:
            print(f"  Warning: Failed to load {audio_path}: {e}")
            skipped_count += 1
            continue

        # Loudness normalize to -23 LUFS
        audio = normalize_loudness(audio, orig_sr, TARGET_LUFS)

        # Resample to 48 kHz
        if orig_sr != TARGET_SR:
            audio = resample_audio(audio, orig_sr, TARGET_SR)

        # Save to processed directory
        filename = Path(audio_path).name
        output_path = MEDLEY_PROCESSED_DIR / instrument_name / filename
        save_audio(output_path, audio, TARGET_SR)

        processed_count += 1

    print(f"\nDone! Processed: {processed_count}, Skipped: {skipped_count}")

    # Print per-instrument counts
    print("\nPer-instrument counts:")
    for inst_name in sorted(INSTRUMENT_MAP.values()):
        inst_dir = MEDLEY_PROCESSED_DIR / inst_name
        count = len(list(inst_dir.glob("*.wav")))
        print(f"  {inst_name}: {count} clips")


def main():
    # Step 1: Download
    dataset = download_medley_solos_db()

    # Step 2: Process
    process_clips(dataset)

    print("\n" + "=" * 60)
    print("Medley-solos-DB download and processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
