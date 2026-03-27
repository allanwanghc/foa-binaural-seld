"""
SELD label encoding for Multi-ACCDOA with ADPIT.

Converts SELD labels from TSV format to Multi-ACCDOA ADPIT format
following the DCASE 2024 baseline approach.

ADPIT (Auxiliary Duplicating Permutation Invariant Training):
  - 6 auxiliary outputs: [a0, b0, b1, c0, c1, c2]
    - a0: all sources assigned to a single track
    - b0, b1: sources split across 2 tracks (for same-class overlap)
    - c0, c1, c2: sources split across 3 tracks (for same-class overlap)
  - Per-frame label shape: (6, 4, num_classes)
    - 4 = [activity, x, y, z]

TSV label format (our dataset):
  frame\tclass_id\tsource_id\tazimuth_deg\televation_deg
  (100 fps, tab-separated, has header row)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def azel_to_cartesian(azimuth_deg, elevation_deg):
    """
    Convert azimuth/elevation in degrees to unit vector (x, y, z).

    Convention follows DCASE baseline:
      x = cos(el) * cos(az)
      y = cos(el) * sin(az)
      z = sin(el)

    Args:
        azimuth_deg: azimuth angle in degrees
        elevation_deg: elevation angle in degrees

    Returns:
        tuple (x, y, z) as floats
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return float(x), float(y), float(z)


def _load_labels_tsv(label_path):
    """
    Load labels from TSV file into a frame-indexed dictionary.

    Our TSV format: frame, class_id, source_id, azimuth_deg, elevation_deg
    (tab-separated, has header)

    Returns:
        dict: {frame_ind: [[class_id, source_id, x, y, z], ...]}
        Frame dict with Cartesian coordinates (matching DCASE baseline format).
    """
    label_path = Path(label_path)
    df = pd.read_csv(label_path, sep="\t")

    desc_dict = {}
    for _, row in df.iterrows():
        frame_ind = int(row["frame"])
        class_id = int(row["class_id"])
        source_id = int(row["source_id"])
        az_deg = float(row["azimuth_deg"])
        el_deg = float(row["elevation_deg"])

        x, y, z = azel_to_cartesian(az_deg, el_deg)

        if frame_ind not in desc_dict:
            desc_dict[frame_ind] = []
        desc_dict[frame_ind].append([class_id, source_id, x, y, z])

    return desc_dict


def seld_labels_to_adpit(label_path, num_frames, num_classes=8):
    """
    Convert SELD labels TSV to ADPIT format for Multi-ACCDOA training.

    Follows the DCASE 2024 baseline `get_adpit_labels_for_file()` approach:
    - Events are sorted by class_id per frame
    - Same-class overlapping sources are routed to different ADPIT tracks:
        * 1 instance of a class -> track a0
        * 2 instances of a class -> tracks b0, b1
        * 3+ instances of a class -> tracks c0, c1, c2

    Args:
        label_path: path to labels.tsv file
        num_frames: number of feature frames (determines output length)
        num_classes: number of sound event classes (default 8)

    Returns:
        np.ndarray of shape (num_frames, 6, 4, num_classes), dtype float32
          6 = ADPIT auxiliary tracks [a0, b0, b1, c0, c1, c2]
          4 = [activity, x, y, z]
          num_classes = 8 instrument classes
    """
    desc_dict = _load_labels_tsv(label_path)

    # Initialize label arrays
    se_label = np.zeros((num_frames, 6, num_classes), dtype=np.float32)
    x_label = np.zeros((num_frames, 6, num_classes), dtype=np.float32)
    y_label = np.zeros((num_frames, 6, num_classes), dtype=np.float32)
    z_label = np.zeros((num_frames, 6, num_classes), dtype=np.float32)

    for frame_ind, active_event_list in desc_dict.items():
        if frame_ind >= num_frames:
            continue

        # Sort by class_id to group same-class events together
        active_event_list.sort(key=lambda ev: ev[0])

        # Process events grouped by class (following DCASE baseline logic)
        events_per_class = []
        for i, event in enumerate(active_event_list):
            events_per_class.append(event)

            # Check if this is the last event or next event is a different class
            is_last = (i == len(active_event_list) - 1)
            is_class_boundary = (
                not is_last and event[0] != active_event_list[i + 1][0]
            )

            if is_last or is_class_boundary:
                n_same_class = len(events_per_class)

                if n_same_class == 1:
                    # Single instance -> track a0 (index 0)
                    ev = events_per_class[0]
                    cid = ev[0]
                    se_label[frame_ind, 0, cid] = 1
                    x_label[frame_ind, 0, cid] = ev[2]
                    y_label[frame_ind, 0, cid] = ev[3]
                    z_label[frame_ind, 0, cid] = ev[4]

                elif n_same_class == 2:
                    # 2 same-class instances -> tracks b0, b1 (indices 1, 2)
                    ev0 = events_per_class[0]
                    ev1 = events_per_class[1]
                    cid = ev0[0]

                    se_label[frame_ind, 1, cid] = 1
                    x_label[frame_ind, 1, cid] = ev0[2]
                    y_label[frame_ind, 1, cid] = ev0[3]
                    z_label[frame_ind, 1, cid] = ev0[4]

                    se_label[frame_ind, 2, cid] = 1
                    x_label[frame_ind, 2, cid] = ev1[2]
                    y_label[frame_ind, 2, cid] = ev1[3]
                    z_label[frame_ind, 2, cid] = ev1[4]

                else:
                    # 3+ same-class instances -> tracks c0, c1, c2 (indices 3, 4, 5)
                    # Only first 3 are kept (MAX_TRACKS = 3)
                    for track_idx, ev in enumerate(events_per_class[:3]):
                        cid = ev[0]
                        se_label[frame_ind, 3 + track_idx, cid] = 1
                        x_label[frame_ind, 3 + track_idx, cid] = ev[2]
                        y_label[frame_ind, 3 + track_idx, cid] = ev[3]
                        z_label[frame_ind, 3 + track_idx, cid] = ev[4]

                # Reset for next class group
                events_per_class = []

    # Stack: (num_frames, 6, 4, num_classes)
    # Order: [activity, x, y, z] matching DCASE baseline
    label_mat = np.stack(
        (se_label, x_label, y_label, z_label), axis=2
    )

    return label_mat.astype(np.float32)
