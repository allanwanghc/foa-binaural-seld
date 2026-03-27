"""
Quality control checks for generated scenes.

1. Energy check: Verify FOA and binaural audio have sufficient energy
2. Channel count: FOA=4ch, Binaural=2ch
3. Label consistency: Labels match metadata
4. Duration check: Audio duration matches scene plan
5. Spatial: FOA active intensity direction vs label direction
6. Spatial: Binaural ILD sign vs label azimuth sign
7. Spatial: Planned vs rendered DOA snap error
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path


def check_audio_energy(filepath, threshold_dbfs=-45.0):
    """Check that audio has sufficient energy (not silent)."""
    audio, sr = sf.read(str(filepath))
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        dbfs = -np.inf
    else:
        dbfs = 20 * np.log10(rms)
    return dbfs > threshold_dbfs, dbfs


def check_foa_channels(filepath):
    """Verify FOA audio has exactly 4 channels."""
    info = sf.info(str(filepath))
    return info.channels == 4, info.channels


def check_binaural_channels(filepath):
    """Verify binaural audio has exactly 2 channels."""
    info = sf.info(str(filepath))
    return info.channels == 2, info.channels


def check_sample_rate(filepath, expected_sr):
    """Verify sample rate matches expected."""
    info = sf.info(str(filepath))
    return info.samplerate == expected_sr, info.samplerate


def check_duration(filepath, expected_duration, tolerance=0.5):
    """Verify audio duration is within tolerance of expected."""
    info = sf.info(str(filepath))
    actual_duration = info.duration
    return abs(actual_duration - expected_duration) < tolerance, actual_duration


def check_label_count(label_path, meta_path):
    """Verify labels contain expected number of sources."""
    with open(meta_path) as f:
        meta = json.load(f)
    expected_sources = meta["num_sources"]

    # Read labels
    source_ids = set()
    with open(label_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                source_ids.add(int(parts[2]))

    return len(source_ids) == expected_sources, (len(source_ids), expected_sources)


def _read_labels(label_path):
    """Read labels.tsv and return list of (frame, class_id, source_id, az, el)."""
    rows = []
    with open(label_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2]),
                             float(parts[3]), float(parts[4])))
    return rows


def great_circle_distance(az1, el1, az2, el2):
    """Angular distance in degrees between two (az, el) directions."""
    az1, el1, az2, el2 = map(np.radians, [az1, el1, az2, el2])
    cos_dist = (np.sin(el1) * np.sin(el2) +
                np.cos(el1) * np.cos(el2) * np.cos(az1 - az2))
    return np.degrees(np.arccos(np.clip(cos_dist, -1, 1)))


def check_foa_intensity_direction(foa_path, label_path, threshold_deg=60.0):
    """Check FOA active intensity direction vs label direction."""
    audio, sr = sf.read(str(foa_path))  # (N, 4) = W, Y, Z, X
    W, Y, Z, X = audio[:, 0], audio[:, 1], audio[:, 2], audio[:, 3]
    Ix, Iy, Iz = np.mean(W * X), np.mean(W * Y), np.mean(W * Z)
    intensity_mag = np.sqrt(Ix**2 + Iy**2 + Iz**2)
    if intensity_mag < 1e-10:
        return True, "skip (silent)"
    az_foa = np.degrees(np.arctan2(Iy, Ix))
    el_foa = np.degrees(np.arcsin(np.clip(Iz / intensity_mag, -1, 1)))

    rows = _read_labels(label_path)
    if not rows:
        return True, "skip (no labels)"
    az_label = np.mean([r[3] for r in rows])
    el_label = np.mean([r[4] for r in rows])

    angular_error = great_circle_distance(az_foa, el_foa, az_label, el_label)
    return angular_error < threshold_deg, round(angular_error, 1)


def check_binaural_ild_consistency(binaural_path, label_path, min_az=15.0):
    """Check binaural ILD sign matches label azimuth sign."""
    audio, sr = sf.read(str(binaural_path))  # (N, 2)
    rms_l = np.sqrt(np.mean(audio[:, 0] ** 2))
    rms_r = np.sqrt(np.mean(audio[:, 1] ** 2))
    ild = 20 * np.log10(max(rms_l, 1e-10) / max(rms_r, 1e-10))

    rows = _read_labels(label_path)
    if not rows:
        return True, "skip (no labels)"
    mean_az = np.mean([r[3] for r in rows])

    if abs(mean_az) < min_az:
        return True, f"skip (|az|={abs(mean_az):.1f}< {min_az})"

    consistent = (mean_az > 0 and ild > 0) or (mean_az < 0 and ild < 0)
    return consistent, f"ILD={ild:.2f}dB, mean_az={mean_az:.1f}"


def check_doa_snap_error(meta_path, threshold_deg=20.0):
    """Check planned vs rendered DOA angular error."""
    with open(meta_path) as f:
        meta = json.load(f)
    errors = []
    for src in meta["sources"]:
        if "planned_azimuth_deg" not in src:
            continue
        err = great_circle_distance(
            src["azimuth_deg"], src["elevation_deg"],
            src["planned_azimuth_deg"], src["planned_elevation_deg"])
        errors.append(err)
    if not errors:
        return True, "skip (no planned DOA)"
    mean_err = np.mean(errors)
    return mean_err < threshold_deg, round(mean_err, 1)


def run_qc_scene(scene_dir, expected_sr=48000):
    """
    Run all QC checks on a single scene.

    Returns:
        dict with check names as keys and (passed, detail) as values
    """
    scene_dir = Path(scene_dir)
    results = {}

    foa_path = scene_dir / "foa_WXYZ.wav"
    binaural_path = scene_dir / "binaural_LR.wav"
    meta_path = scene_dir / "meta.json"
    label_path = scene_dir / "labels.tsv"

    # Check files exist
    for name, path in [("foa_exists", foa_path), ("binaural_exists", binaural_path),
                        ("meta_exists", meta_path), ("labels_exists", label_path)]:
        results[name] = (path.exists(), str(path))

    if not all(r[0] for r in results.values()):
        return results

    # Load meta for duration check
    with open(meta_path) as f:
        meta = json.load(f)

    # FOA checks
    results["foa_channels"] = check_foa_channels(foa_path)
    results["foa_sr"] = check_sample_rate(foa_path, expected_sr)
    results["foa_energy"] = check_audio_energy(foa_path)
    results["foa_duration"] = check_duration(foa_path, meta["duration"])

    # Binaural checks
    results["binaural_channels"] = check_binaural_channels(binaural_path)
    results["binaural_sr"] = check_sample_rate(binaural_path, expected_sr)
    results["binaural_energy"] = check_audio_energy(binaural_path)

    # Label checks
    results["label_sources"] = check_label_count(label_path, meta_path)

    # Spatial checks
    results["spatial_foa_intensity"] = check_foa_intensity_direction(foa_path, label_path)
    results["spatial_ild_consistency"] = check_binaural_ild_consistency(binaural_path, label_path)
    results["spatial_doa_snap"] = check_doa_snap_error(meta_path)

    return results
