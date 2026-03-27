"""
Binaural feature extraction.

Extracts spatial audio features from 2-channel binaural audio:
  - Log-mel spectrograms for L and R channels
  - ILD (Interaural Level Difference)
  - IPD (Interaural Phase Difference) as cos and sin components
  - GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

Output shape: (T, 6, n_mels)
  Channels: [mel_L, mel_R, ILD, IPD_cos, IPD_sin, GCC]
"""

import numpy as np
import librosa


def extract_binaural_features(
    audio_path,
    sr=48000,
    n_fft=2048,
    hop_length=480,
    n_mels=64,
    fmin=50,
    fmax=22000,
):
    """
    Extract binaural features from a 2-channel binaural audio file.

    Args:
        audio_path: path to 2-channel binaural .wav file (L, R)
        sr: sample rate
        n_fft: FFT size
        hop_length: hop length in samples (48000/480 = 100 fps)
        n_mels: number of mel bins
        fmin: minimum frequency for mel filterbank
        fmax: maximum frequency for mel filterbank

    Returns:
        np.ndarray of shape (T, 6, n_mels), dtype float32
          channels = [mel_L, mel_R, ILD, IPD_cos, IPD_sin, GCC]
    """
    eps = 1e-8

    # Load 2-channel audio
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=False)
    # audio shape: (2, num_samples)
    assert audio.ndim == 2 and audio.shape[0] == 2, (
        f"Expected 2-channel audio, got shape {audio.shape}"
    )

    # Compute STFT for each channel
    stft_L = librosa.stft(
        audio[0], n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window="hann",
    )  # (n_freq, T)
    stft_R = librosa.stft(
        audio[1], n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, window="hann",
    )  # (n_freq, T)

    # Transpose to (T, n_freq)
    stft_L = stft_L.T  # (T, n_freq)
    stft_R = stft_R.T  # (T, n_freq)
    num_frames = stft_L.shape[0]

    # Build mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    ).T  # (n_freq, n_mels)

    # --- Log-mel spectrograms ---
    mag_L_sq = np.abs(stft_L) ** 2  # (T, n_freq)
    mag_R_sq = np.abs(stft_R) ** 2

    mel_L = np.dot(mag_L_sq, mel_basis)  # (T, n_mels)
    mel_R = np.dot(mag_R_sq, mel_basis)

    log_mel_L = librosa.power_to_db(mel_L)  # (T, n_mels)
    log_mel_R = librosa.power_to_db(mel_R)

    # --- ILD (Interaural Level Difference) ---
    # ILD = 20 * log10(|R_mel| / (|L_mel| + eps))
    # Use sqrt of mel power for magnitude
    mel_L_mag = np.sqrt(mel_L + eps)
    mel_R_mag = np.sqrt(mel_R + eps)
    ild = 20.0 * np.log10(mel_R_mag / (mel_L_mag + eps))  # (T, n_mels)

    # --- IPD (Interaural Phase Difference) ---
    # Cross-spectrum: L * conj(R)
    cross_spec = stft_L * np.conj(stft_R)  # (T, n_freq)
    phase_diff = np.angle(cross_spec)  # (T, n_freq)

    # Aggregate to mel bins: weight by mel filterbank
    # cos(phase) and sin(phase) projected to mel
    cos_phase = np.cos(phase_diff)  # (T, n_freq)
    sin_phase = np.sin(phase_diff)  # (T, n_freq)

    ipd_cos = np.dot(cos_phase, mel_basis)  # (T, n_mels)
    ipd_sin = np.dot(sin_phase, mel_basis)  # (T, n_mels)

    # Normalize by number of freq bins contributing to each mel bin
    mel_bin_counts = mel_basis.sum(axis=0, keepdims=True)  # (1, n_mels)
    mel_bin_counts = np.maximum(mel_bin_counts, eps)
    ipd_cos = ipd_cos / mel_bin_counts
    ipd_sin = ipd_sin / mel_bin_counts

    # --- GCC-PHAT ---
    # GCC-PHAT = IFFT(L * conj(R) / |L * conj(R)|)
    cross_spec_full = stft_L * np.conj(stft_R)  # (T, n_freq)
    cross_mag = np.abs(cross_spec_full) + eps
    gcc_phat_spec = cross_spec_full / cross_mag  # (T, n_freq)

    # Inverse FFT to get GCC-PHAT in time domain, then project to mel bins
    gcc_phat_td = np.fft.irfft(gcc_phat_spec, axis=-1)  # (T, n_fft)
    # Take center portion matching n_mels
    gcc_centered = np.concatenate(
        [gcc_phat_td[:, -n_mels // 2:], gcc_phat_td[:, :n_mels // 2]],
        axis=-1,
    )  # (T, n_mels)

    # --- Stack all features ---
    # Shape: (T, 6, n_mels)
    features = np.stack(
        [log_mel_L, log_mel_R, ild, ipd_cos, ipd_sin, gcc_centered],
        axis=1,
    )

    return features.astype(np.float32)
