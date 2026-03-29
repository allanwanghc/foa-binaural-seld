"""
Central configuration for the thesis dataset generation pipeline.
All scripts import constants from here.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCENES_DIR = OUTPUT_DIR / "scenes"

# Medley-solos-DB
MEDLEY_RAW_DIR = DATA_DIR / "medley_solos_db" / "raw"
MEDLEY_PROCESSED_DIR = DATA_DIR / "medley_solos_db" / "processed"

# HRTFs
HRTF_DIR = DATA_DIR / "hrtfs"
HRTF_FILES = {
    "ku100": HRTF_DIR / "ku100" / "HRIR_FULL2DEG.sofa",
    "cipic": HRTF_DIR / "cipic" / "subject_003.sofa",
    "sadie_ii": HRTF_DIR / "sadie_ii" / "H3_HRIR_SOFA" / "H3_48K_24bit_256tap_FIR_SOFA.sofa",
}

# RIRs (for SpatialScaper)
RIR_DIR = DATA_DIR / "rir_datasets"

# SpatialScaper
SPATIALSCAPER_DIR = PROJECT_ROOT / "SpatialScaper-main"

# ─── Audio Parameters ────────────────────────────────────────────────
TARGET_SR = 48000
TARGET_LUFS = -23.0

# ─── Instrument Classes ──────────────────────────────────────────────
# Medley-solos-DB instrument IDs → names
INSTRUMENT_MAP = {
    0: "clarinet",
    1: "distorted_electric_guitar",
    2: "female_singer",
    3: "flute",
    4: "piano",
    5: "tenor_saxophone",
    6: "trumpet",
    7: "violin",
}
INSTRUMENT_NAMES = list(INSTRUMENT_MAP.values())
NUM_CLASSES = len(INSTRUMENT_MAP)

# ─── Experimental Factors ────────────────────────────────────────────
AZIMUTHS = list(range(-90, 91, 15))       # 13 levels: -90, -75, ..., 75, 90
ELEVATIONS = list(range(-30, 31, 10))     # 7 levels: -30, -20, ..., 20, 30
NUM_SOURCES_RANGE = (2, 3)                # 2-3 sources per scene
DURATION_RANGE = (2.0, 6.0)              # seconds

# Rooms: 3 SpatialScaper rooms with different RT60 (must have FOA RIRs)
# bomb_shelter: large underground space, high RT60
# gym: large open gym, high RT60
# sc203: small classroom with carpet, low RT60
ROOMS = ["bomb_shelter", "gym", "sc203"]

# HRTF sets for binaural rendering
HRTF_SETS = ["ku100", "cipic", "sadie_ii"]

# ─── Dataset Parameters ──────────────────────────────────────────────
NUM_SCENES = 10000
LABEL_FPS = 100  # frames per second for SELD labels

# ─── Quality Control ─────────────────────────────────────────────────
QC_ENERGY_THRESHOLD_DBFS = -45.0  # SpatialScaper uses ref_db=-60, so output is quiet
QC_DIRECTION_THRESHOLD_DEG = 10.0

# ─── Dataset Split ───────────────────────────────────────────────────
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

# ─── Feature Extraction ────────────────────────────────────────
FEATURE_DIR = OUTPUT_DIR / "features"
N_FFT = 2048
HOP_LENGTH = 480          # 48000 / 480 = 100 fps = LABEL_FPS
N_MELS = 64
FMIN = 50
FMAX = 22000

# FOA: 4 log-mel + 3 IV = 7 channels
FOA_NUM_CHANNELS = 7
# Binaural: 2 log-mel + 1 ILD + 2 IPD(cos,sin) + 1 GCC = 6 channels
BIN_NUM_CHANNELS = 6

# Multi-ACCDOA
MAX_TRACKS = 3  # max simultaneous sources per class

# Training
CHUNK_LENGTH_S = 5.0  # match DCASE baseline: 5 second chunks
CHUNK_LENGTH_FRAMES = int(CHUNK_LENGTH_S * LABEL_FPS)  # 500 frames
