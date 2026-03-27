"""
Label format conversion utilities for SELD annotations.

Converts SpatialScaper's XYZ-based labels to azimuth/elevation format
and generates SELD-compatible label files.
"""

import numpy as np
import csv
from pathlib import Path


def xyz_to_azel(x, y, z):
    """
    Convert Cartesian (x, y, z) to (azimuth, elevation) in degrees.

    Convention:
        azimuth: 0 = front (+x), positive = left (+y), range [-180, 180]
        elevation: 0 = horizontal, positive = up (+z), range [-90, 90]
    """
    azimuth = np.degrees(np.arctan2(y, x))
    distance = np.sqrt(x**2 + y**2 + z**2)
    if distance < 1e-10:
        elevation = 0.0
    else:
        elevation = np.degrees(np.arcsin(z / distance))
    return azimuth, elevation


def ss_labels_to_seld(ss_labels, receiver_pos, label_fps=100):
    """
    Convert SpatialScaper label matrix to SELD format.

    SpatialScaper labels: [frame_idx, x, y, z, class_id, source_id]
    SELD labels: [frame_idx, class_id, source_id, azimuth_deg, elevation_deg]

    Args:
        ss_labels: numpy array with SS label format
        receiver_pos: [x, y, z] of receiver in room coordinates
        label_fps: label frame rate

    Returns:
        list of [frame_idx, class_id, source_id, azimuth_deg, elevation_deg]
    """
    seld_labels = []
    rx, ry, rz = receiver_pos

    for row in ss_labels:
        frame_idx = int(row[0])
        sx, sy, sz = row[1], row[2], row[3]
        class_id = int(row[4])
        source_id = int(row[5])

        # Compute relative position
        dx, dy, dz = sx - rx, sy - ry, sz - rz
        azimuth, elevation = xyz_to_azel(dx, dy, dz)

        seld_labels.append([
            frame_idx,
            class_id,
            source_id,
            round(azimuth, 1),
            round(elevation, 1),
        ])

    return seld_labels


def generate_static_seld_labels(scene, label_fps=100):
    """
    Generate SELD labels directly from scene plan for static sources.

    Since all sources are static, we know the exact az/el from the plan.
    Labels are generated for each frame where the source is active.

    Args:
        scene: scene dict from scene_plan.json
        label_fps: frames per second

    Returns:
        list of [frame_idx, class_id, source_id, azimuth_deg, elevation_deg]
    """
    duration = scene["duration"]
    total_frames = int(duration * label_fps)
    labels = []

    for source_idx, source in enumerate(scene["sources"]):
        class_id = source["instrument_class"]
        azimuth = source["azimuth_deg"]
        elevation = source["elevation_deg"]

        # Source is active for all frames (static, full duration)
        # Note: actual activity depends on whether audio content exists
        # This is a "plan" label; real labels come from SpatialScaper
        for frame in range(total_frames):
            labels.append([
                frame,
                class_id,
                source_idx,
                azimuth,
                elevation,
            ])

    # Sort by frame, then class_id
    labels.sort(key=lambda x: (x[0], x[1], x[2]))
    return labels


def save_seld_labels(labels, filepath):
    """
    Save SELD labels to TSV file.

    Format: frame_idx<TAB>class_id<TAB>source_id<TAB>azimuth<TAB>elevation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["frame", "class_id", "source_id", "azimuth", "elevation"])
        for row in labels:
            writer.writerow(row)


def load_seld_labels(filepath):
    """Load SELD labels from TSV file."""
    labels = []
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            labels.append([
                int(row[0]),
                int(row[1]),
                int(row[2]),
                float(row[3]),
                float(row[4]),
            ])
    return labels
