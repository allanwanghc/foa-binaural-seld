"""
Audio utility functions: loudness normalization, resampling, etc.
"""

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln


def load_audio(filepath, sr=None):
    """Load audio file, return (audio, sample_rate)."""
    audio, orig_sr = sf.read(filepath, dtype="float32")
    if sr is not None and orig_sr != sr:
        audio = resample_audio(audio, orig_sr, sr)
        return audio, sr
    return audio, orig_sr


def resample_audio(audio, orig_sr, target_sr):
    """Resample audio from orig_sr to target_sr using librosa."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def measure_loudness(audio, sr):
    """Measure integrated loudness in LUFS (ITU-R BS.1770-4)."""
    meter = pyln.Meter(sr)
    # pyloudnorm expects shape (samples,) for mono
    if audio.ndim == 1:
        return meter.integrated_loudness(audio)
    else:
        return meter.integrated_loudness(audio)


def normalize_loudness(audio, sr, target_lufs=-23.0):
    """
    Loudness-normalize audio to target LUFS.
    Returns normalized audio. If audio is silent, returns as-is.
    """
    current_lufs = measure_loudness(audio, sr)
    if current_lufs == -np.inf or np.isnan(current_lufs):
        # Silent or invalid audio, return as-is
        return audio
    return pyln.normalize.loudness(audio, current_lufs, target_lufs)


def save_audio(filepath, audio, sr):
    """Save audio as WAV, 32-bit float."""
    sf.write(str(filepath), audio, sr, subtype="FLOAT")


def azel_to_room_xyz(azimuth_deg, elevation_deg, distance, receiver_pos):
    """
    Convert (azimuth, elevation, distance) to room XYZ coordinates.

    Convention:
        azimuth: 0 = front (+x), positive = left (+y)
        elevation: 0 = horizontal, positive = up (+z)
        distance: meters from receiver

    Args:
        azimuth_deg: azimuth angle in degrees
        elevation_deg: elevation angle in degrees
        distance: distance from receiver in meters
        receiver_pos: [x, y, z] of receiver in room coordinates

    Returns:
        [x, y, z] source position in room coordinates
    """
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    dx = distance * np.cos(el_rad) * np.cos(az_rad)
    dy = distance * np.cos(el_rad) * np.sin(az_rad)
    dz = distance * np.sin(el_rad)
    return [
        receiver_pos[0] + dx,
        receiver_pos[1] + dy,
        receiver_pos[2] + dz,
    ]
