"""
FOA (First-Order Ambisonics) feature extraction.

Extracts log-mel spectrograms and intensity vectors from 4-channel FOA audio,
following the DCASE SELD baseline approach.

Output shape: (T, 7, n_mels)
  Channels: [mel_W, mel_Y, mel_Z, mel_X, iv_Y, iv_Z, iv_X]
"""

import numpy as np
import librosa


def extract_foa_features(
    audio_path,
    sr=48000,
    n_fft=2048,
    hop_length=480,
    n_mels=64,
    fmin=50,
    fmax=22000,
):
    """
    Extract FOA features from a 4-channel FOA audio file.

    Args:
        audio_path: path to 4-channel FOA .wav file (W, Y, Z, X ordering)
        sr: sample rate
        n_fft: FFT size
        hop_length: hop length in samples (48000/480 = 100 fps)
        n_mels: number of mel bins
        fmin: minimum frequency for mel filterbank
        fmax: maximum frequency for mel filterbank

    Returns:
        np.ndarray of shape (T, 7, n_mels), dtype float32
          channels = [mel_W, mel_Y, mel_Z, mel_X, iv_Y, iv_Z, iv_X]
    """
    # Load 4-channel audio
    audio, _ = librosa.load(str(audio_path), sr=sr, mono=False)
    # audio shape: (4, num_samples)
    assert audio.ndim == 2 and audio.shape[0] == 4, (
        f"Expected 4-channel audio, got shape {audio.shape}"
    )

    nb_ch = audio.shape[0]

    # Compute STFT for each channel
    # Using librosa for consistency with DCASE baseline
    stft_all = []
    for ch in range(nb_ch):
        stft_ch = librosa.stft(
            audio[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
        )
        stft_all.append(stft_ch)
    # stft_all: list of 4 arrays, each (n_fft//2+1, T)

    # Stack and transpose to (T, n_freq, n_ch)
    linear_spectra = np.array(stft_all).transpose(2, 1, 0)  # (T, n_freq, 4)
    num_frames = linear_spectra.shape[0]

    # Build mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    ).T  # (n_freq, n_mels)

    # --- Log-mel spectrograms for all 4 channels ---
    mel_specs = np.zeros((num_frames, nb_ch, n_mels), dtype=np.float32)
    for ch in range(nb_ch):
        mag_sq = np.abs(linear_spectra[:, :, ch]) ** 2  # (T, n_freq)
        mel_spec = np.dot(mag_sq, mel_basis)  # (T, n_mels)
        mel_specs[:, ch, :] = librosa.power_to_db(mel_spec)

    # --- FOA Intensity Vectors ---
    # IV_i = Re(conj(W) * Ch_i) for i in {Y, Z, X} (channels 1, 2, 3)
    eps = 1e-8
    W = linear_spectra[:, :, 0]  # (T, n_freq)
    I = np.real(
        np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:]
    )  # (T, n_freq, 3)

    # Energy normalization (following DCASE baseline)
    E = eps + np.abs(W) ** 2 + (np.abs(linear_spectra[:, :, 1:]) ** 2).sum(-1) / 3.0
    # E shape: (T, n_freq)

    I_norm = I / E[:, :, np.newaxis]  # (T, n_freq, 3)

    # Project to mel bins
    # I_norm: (T, n_freq, 3) -> transpose to (T, 3, n_freq), dot with mel_basis
    I_norm_mel = np.dot(
        I_norm.transpose(0, 2, 1), mel_basis
    ).transpose(0, 2, 1)
    # After: (T, n_mels, 3) -> transpose back to (T, 3, n_mels)
    iv_features = I_norm_mel.transpose(0, 2, 1)  # (T, 3, n_mels)

    if np.isnan(iv_features).any():
        raise ValueError("FOA intensity vector extraction produced NaN values")

    # Concatenate: [mel_W, mel_Y, mel_Z, mel_X, iv_Y, iv_Z, iv_X]
    features = np.concatenate([mel_specs, iv_features], axis=1)  # (T, 7, n_mels)

    return features.astype(np.float32)
