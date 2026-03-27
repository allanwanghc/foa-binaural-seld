#!/usr/bin/env python3
"""
Step 1b: Process already-extracted Medley-solos-DB files.
Reads metadata CSV, matches to extracted WAV files,
loudness-normalizes to -23 LUFS, resamples to 48 kHz,
and saves organized by instrument subdirectory.

Output structure:
    data/medley_solos_db/processed/{instrument_name}/*.wav
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

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


# Map instrument names to directory-safe names
INSTRUMENT_DIR_NAMES = {
    "clarinet": "clarinet",
    "distorted electric guitar": "distorted_electric_guitar",
    "female singer": "female_singer",
    "flute": "flute",
    "piano": "piano",
    "tenor saxophone": "tenor_saxophone",
    "trumpet": "trumpet",
    "violin": "violin",
}


def main():
    audio_dir = MEDLEY_RAW_DIR / "audio"
    metadata_path = MEDLEY_RAW_DIR / "annotation" / "Medley-solos-DB_metadata.csv"

    print("=" * 60)
    print("Processing Medley-solos-DB")
    print(f"  Audio dir: {audio_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Output: {MEDLEY_PROCESSED_DIR}")
    print(f"  Target LUFS: {TARGET_LUFS}")
    print(f"  Target SR: {TARGET_SR} Hz")
    print("=" * 60)

    # Read metadata
    df = pd.read_csv(metadata_path)
    print(f"\nTotal clips in metadata: {len(df)}")

    # Build expected filename
    df["filename"] = df.apply(
        lambda r: f"Medley-solos-DB_{r['subset']}-{r['instrument_id']}_{r['uuid4']}.wav",
        axis=1,
    )
    df["filepath"] = df["filename"].apply(lambda f: audio_dir / f)
    df["exists"] = df["filepath"].apply(lambda f: f.exists())

    available = df[df["exists"]]
    print(f"Available audio files: {len(available)}")
    print(f"Missing: {len(df) - len(available)}")

    # Create output subdirectories
    for dir_name in INSTRUMENT_DIR_NAMES.values():
        (MEDLEY_PROCESSED_DIR / dir_name).mkdir(parents=True, exist_ok=True)

    # Process each file
    processed = 0
    errors = 0

    for _, row in tqdm(available.iterrows(), total=len(available), desc="Processing"):
        instrument = row["instrument"]
        dir_name = INSTRUMENT_DIR_NAMES[instrument]
        input_path = row["filepath"]
        output_path = MEDLEY_PROCESSED_DIR / dir_name / row["filename"]

        # Skip if already processed
        if output_path.exists():
            processed += 1
            continue

        try:
            # Load audio
            audio, orig_sr = load_audio(str(input_path))

            # Loudness normalize
            audio = normalize_loudness(audio, orig_sr, TARGET_LUFS)

            # Resample to 48 kHz
            if orig_sr != TARGET_SR:
                audio = resample_audio(audio, orig_sr, TARGET_SR)

            # Save
            save_audio(output_path, audio, TARGET_SR)
            processed += 1

        except Exception as e:
            print(f"\n  Error processing {row['filename']}: {e}")
            errors += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"\nPer-instrument counts:")
    for dir_name in sorted(INSTRUMENT_DIR_NAMES.values()):
        inst_dir = MEDLEY_PROCESSED_DIR / dir_name
        count = len(list(inst_dir.glob("*.wav")))
        print(f"  {dir_name}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
