"""
Binaural renderer: FOA → Binaural using SH-domain HRTF decoding.

Converts 4-channel First-Order Ambisonics (ACN/N3D: W, Y, Z, X)
to 2-channel binaural audio using Head-Related Transfer Functions.

The approach:
1. Load HRTF SOFA file
2. Compute SH coefficients (order 1) for each HRTF measurement direction
3. Build 4 decoding filters per ear by weighted summation of HRIRs
4. Convolve each FOA channel with corresponding decoder filter and sum
"""

import numpy as np
from scipy import signal
import librosa
import pysofaconventions as pysofa


def sph2cart(az_rad, el_rad):
    """Spherical to Cartesian (unit sphere)."""
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)
    return x, y, z


def real_sh_n3d(order, az_rad, el_rad):
    """
    Compute real spherical harmonics up to given order, N3D normalization.

    For first order (order=1), returns 4 coefficients per direction:
        Y_00 = 1                    (W - omnidirectional)
        Y_1-1 = sin(az)*cos(el)    (Y - left/right)
        Y_10 = sin(el)             (Z - up/down)
        Y_11 = cos(az)*cos(el)     (X - front/back)

    ACN ordering: [0,0], [1,-1], [1,0], [1,1] = W, Y, Z, X

    Args:
        order: max SH order (1 for FOA)
        az_rad: azimuth in radians (N x 1)
        el_rad: elevation in radians (N x 1)

    Returns:
        Y: (N, (order+1)^2) SH coefficients, N3D normalized
    """
    n_dirs = len(az_rad)
    n_coeffs = (order + 1) ** 2
    Y = np.zeros((n_dirs, n_coeffs))

    # Order 0: Y_00
    Y[:, 0] = 1.0  # N3D: sqrt(1) = 1

    if order >= 1:
        # Order 1 with N3D normalization (sqrt(3) factor)
        Y[:, 1] = np.sqrt(3) * np.sin(az_rad) * np.cos(el_rad)   # Y_1-1 (Y)
        Y[:, 2] = np.sqrt(3) * np.sin(el_rad)                      # Y_10  (Z)
        Y[:, 3] = np.sqrt(3) * np.cos(az_rad) * np.cos(el_rad)   # Y_11  (X)

    return Y


class BinauralRenderer:
    """
    SH-domain HRTF decoder for FOA → Binaural conversion.

    Precomputes decoding filters from a SOFA HRTF file.
    Each call to render() convolves 4-channel FOA with the filters.
    """

    def __init__(self, sofa_path, target_sr=48000):
        """
        Initialize renderer from a SOFA HRTF file.

        Args:
            sofa_path: path to SOFA file (SimpleFreeFieldHRIR)
            target_sr: target sample rate (will resample HRIRs if needed)
        """
        self.target_sr = target_sr
        self.sofa_path = str(sofa_path)

        # Load SOFA
        sofa = pysofa.SOFAFile(self.sofa_path, "r")
        assert sofa.isValid(), f"Invalid SOFA file: {sofa_path}"

        # Get data
        hrir_data = np.array(sofa.getDataIR())        # (M, R, N) - measurements, receivers, samples
        source_pos = np.array(sofa.getVariableValue("SourcePosition"))  # (M, 3) - az, el, dist
        sr_sofa = float(np.array(sofa.getVariableValue("Data.SamplingRate"))[0])

        n_meas, n_ears, n_samples = hrir_data.shape
        assert n_ears == 2, f"Expected 2 ears, got {n_ears}"

        print(f"  HRTF loaded: {n_meas} measurements, {n_samples} samples, {sr_sofa} Hz")

        # Convert SOFA source positions to radians
        # SOFA convention: azimuth in degrees, elevation in degrees, distance in meters
        az_rad = np.radians(source_pos[:, 0])
        el_rad = np.radians(source_pos[:, 1])

        # Compute SH matrix for all measurement directions
        # Y: (M, 4) for first-order SH
        Y = real_sh_n3d(order=1, az_rad=az_rad, el_rad=el_rad)

        # Compute SH-domain HRTF filters via pseudo-inverse
        # For each ear, solve: H(m, n) ≈ Σ_l Y_l(dir_m) * h_l(n)
        # h_l = (Y^T Y)^-1 Y^T H  (least-squares fit)
        #
        # Y: (M, 4), H: (M, N) for each ear
        # h: (4, N) decode filters for each ear

        YtY_inv = np.linalg.pinv(Y.T @ Y)  # (4, 4)
        YtY_inv_Yt = YtY_inv @ Y.T          # (4, M)

        self.decode_filters = np.zeros((2, 4, n_samples))  # (ears, sh_channels, ir_samples)

        for ear in range(2):
            H = hrir_data[:, ear, :]  # (M, N)
            self.decode_filters[ear] = YtY_inv_Yt @ H  # (4, N)

        # Resample decode filters if needed
        if sr_sofa != target_sr:
            print(f"  Resampling HRTF decode filters: {sr_sofa} -> {target_sr} Hz")
            resampled = []
            for ear in range(2):
                ear_filters = []
                for ch in range(4):
                    resampled_filter = librosa.resample(
                        self.decode_filters[ear, ch],
                        orig_sr=sr_sofa,
                        target_sr=target_sr,
                    )
                    ear_filters.append(resampled_filter)
                resampled.append(ear_filters)
            # Use length from first filter (all should be very close)
            new_n_samples = len(resampled[0][0])
            new_filters = np.zeros((2, 4, new_n_samples))
            for ear in range(2):
                for ch in range(4):
                    filt = resampled[ear][ch]
                    new_filters[ear, ch, :len(filt)] = filt[:new_n_samples]
            self.decode_filters = new_filters

        self.filter_len = self.decode_filters.shape[2]
        print(f"  Decode filters: {self.decode_filters.shape} "
              f"({self.filter_len} samples at {target_sr} Hz)")

    def render(self, foa_audio):
        """
        Render FOA audio to binaural.

        Args:
            foa_audio: (n_samples, 4) array, channels in ACN order [W, Y, Z, X]

        Returns:
            binaural: (n_samples, 2) array [Left, Right]
        """
        assert foa_audio.ndim == 2 and foa_audio.shape[1] == 4, \
            f"Expected (n_samples, 4), got {foa_audio.shape}"

        n_samples = foa_audio.shape[0]
        binaural = np.zeros((n_samples, 2))

        for ear in range(2):
            for ch in range(4):
                # Convolve FOA channel with corresponding decode filter
                convolved = signal.fftconvolve(
                    foa_audio[:, ch],
                    self.decode_filters[ear, ch],
                    mode="full",
                )
                binaural[:n_samples, ear] += convolved[:n_samples]

        return binaural


def create_renderers(hrtf_files, target_sr=48000):
    """
    Pre-initialize BinauralRenderers for all HRTF sets.

    Args:
        hrtf_files: dict mapping hrtf_name -> sofa_path
        target_sr: target sample rate

    Returns:
        dict mapping hrtf_name -> BinauralRenderer
    """
    renderers = {}
    for name, path in hrtf_files.items():
        print(f"\nInitializing renderer: {name}")
        renderers[name] = BinauralRenderer(path, target_sr)
    return renderers
