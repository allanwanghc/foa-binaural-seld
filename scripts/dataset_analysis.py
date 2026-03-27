#!/usr/bin/env python3
"""
Dataset analysis script for thesis results section.
Generates 4 publication-quality figures from the rendered dataset.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/Users/awhc0813/Desktop/Thesis Code")
OUTPUT = ROOT / "output"
SCENES_DIR = OUTPUT / "scenes"
FIG_DIR = OUTPUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENE_PLAN = OUTPUT / "scene_plan.json"
QC_REPORT  = OUTPUT / "qc_report.csv"
SPLITS     = OUTPUT / "splits.json"

# ── style ──────────────────────────────────────────────────────────────────
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

ROOM_COLORS = {'bomb_shelter': '#4C72B0', 'gym': '#DD8452', 'sc203': '#55A868'}


# ── helpers ────────────────────────────────────────────────────────────────
def great_circle_deg(az1, el1, az2, el2):
    """Angular distance on unit sphere, inputs in degrees."""
    az1, el1, az2, el2 = map(np.radians, [az1, el1, az2, el2])
    cos_d = (np.sin(el1) * np.sin(el2) +
             np.cos(el1) * np.cos(el2) * np.cos(az1 - az2))
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.degrees(np.arccos(cos_d))


def load_scene_plan():
    print("Loading scene_plan.json ...")
    with open(SCENE_PLAN) as f:
        return json.load(f)


def load_qc():
    print("Loading qc_report.csv ...")
    return pd.read_csv(QC_REPORT)


def load_meta_batch():
    """Read all meta.json files with progress reporting."""
    scene_dirs = sorted([d for d in SCENES_DIR.iterdir() if d.is_dir()])
    total = len(scene_dirs)
    records = []
    print(f"Reading {total} meta.json files ...")
    for i, sd in enumerate(scene_dirs):
        meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            records.append(json.load(f))
        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"  ... {i+1}/{total}")
    print(f"Loaded {len(records)} meta files.")
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – Dataset Statistics Summary Table
# ═══════════════════════════════════════════════════════════════════════════
def figure1(plan, qc, splits):
    print("\n=== Figure 1: Dataset Statistics Summary ===")

    rooms = Counter(s['room'] for s in plan)
    hrtfs = Counter(s['hrtf_set'] for s in plan)
    n_sources = Counter(s['num_sources'] for s in plan)
    instruments = Counter()
    for s in plan:
        for src in s['sources']:
            instruments[src['instrument_name']] += 1

    total_dur = sum(s['duration'] for s in plan)

    data = [
        ["Total scenes",          f"{len(plan):,}"],
        ["Total duration",        f"{total_dur/3600:.1f} h  ({total_dur:.0f} s)"],
        ["Sample rate",           "48 kHz"],
        ["Audio formats",         "4-ch FOA (AmbiX) + 2-ch binaural"],
        ["Label frame rate",      "100 fps"],
        ["Train / Val / Test",    f"{len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}"],
        ["Rooms",                 ", ".join(f"{k} ({v})" for k, v in sorted(rooms.items()))],
        ["HRTF sets",             ", ".join(f"{k} ({v})" for k, v in sorted(hrtfs.items()))],
        ["Sources per scene",     ", ".join(f"{k}src ({v})" for k, v in sorted(n_sources.items()))],
        ["Instrument classes",    f"{len(instruments)} total"],
    ]
    # Add per-instrument counts
    for inst, cnt in sorted(instruments.items(), key=lambda x: -x[1]):
        data.append([f"  - {inst}", f"{cnt:,}"])

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(data) + 1.0))
    ax.axis('off')
    ax.set_title("Dataset Statistics", fontsize=14, fontweight='bold', pad=12)

    table = ax.table(
        cellText=data,
        colLabels=["Property", "Value"],
        colWidths=[0.38, 0.62],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row shading
    for i in range(1, len(data) + 1):
        for j in range(2):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')

    _save(fig, "fig1_dataset_statistics")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – DOA Snap Error Distribution
# ═══════════════════════════════════════════════════════════════════════════
def figure2(metas):
    print("\n=== Figure 2: DOA Snap Error Distribution ===")

    errors_all = []
    errors_by_room = {}

    for m in metas:
        room = m['room']
        if room not in errors_by_room:
            errors_by_room[room] = []
        for src in m['sources']:
            if 'planned_azimuth_deg' not in src:
                continue
            err = great_circle_deg(
                src['azimuth_deg'], src['elevation_deg'],
                src['planned_azimuth_deg'], src['planned_elevation_deg'],
            )
            errors_all.append(err)
            errors_by_room[room].append(err)

    errors_all = np.array(errors_all)
    mean_err = np.mean(errors_all)
    med_err  = np.median(errors_all)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall histogram
    ax = axes[0]
    ax.hist(errors_all, bins=80, color='#4C72B0', edgecolor='white', alpha=0.85)
    ax.axvline(mean_err, color='red', ls='--', lw=1.5, label=f'Mean = {mean_err:.1f}\u00b0')
    ax.axvline(med_err, color='orange', ls='--', lw=1.5, label=f'Median = {med_err:.1f}\u00b0')
    ax.set_xlabel("Angular Error (\u00b0)")
    ax.set_ylabel("Count (sources)")
    ax.set_title("DOA Snap Error Distribution (all sources)")
    ax.legend(fontsize=10)

    # Right: by room
    ax = axes[1]
    for room in sorted(errors_by_room.keys()):
        arr = np.array(errors_by_room[room])
        ax.hist(arr, bins=60, alpha=0.55, label=f"{room} (n={len(arr)}, med={np.median(arr):.1f}\u00b0)",
                color=ROOM_COLORS.get(room, None), edgecolor='white')
    ax.set_xlabel("Angular Error (\u00b0)")
    ax.set_ylabel("Count (sources)")
    ax.set_title("DOA Snap Error by Room")
    ax.legend(fontsize=9)

    fig.suptitle("Planned vs. Rendered Direction-of-Arrival Error", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_doa_error_distribution")

    print(f"  Overall: mean={mean_err:.2f}\u00b0, median={med_err:.2f}\u00b0, "
          f"max={np.max(errors_all):.2f}\u00b0, <5\u00b0={100*np.mean(errors_all<5):.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – Spatial QC Pass Rates
# ═══════════════════════════════════════════════════════════════════════════
def figure3(qc):
    print("\n=== Figure 3: Spatial QC Pass Rates ===")

    checks = ['foa_energy', 'binaural_energy', 'spatial_foa_intensity',
              'spatial_ild_consistency', 'spatial_doa_snap']
    labels = ['FOA Energy', 'Binaural Energy', 'FOA Intensity',
              'ILD Consistency', 'DOA Snap']

    rates = []
    for c in checks:
        passed = (qc[c] == 'PASS').sum()
        rates.append(100.0 * passed / len(qc))

    colors = ['#55A868' if r >= 95 else '#DD8452' if r >= 85 else '#C44E52' for r in rates]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, rates, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pass Rate (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Spatial Quality Control Pass Rates", fontsize=14, fontweight='bold')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f'{rate:.1f}%', va='center', fontsize=11, fontweight='bold')

    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig3_qc_pass_rates")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Per-Room and Per-HRTF Analysis
# ═══════════════════════════════════════════════════════════════════════════
def figure4(metas, qc, plan):
    print("\n=== Figure 4: Per-Room & Per-HRTF Analysis ===")

    # ── top: DOA snap error by room (box plot) ──
    errors_by_room = {}
    for m in metas:
        room = m['room']
        if room not in errors_by_room:
            errors_by_room[room] = []
        for src in m['sources']:
            if 'planned_azimuth_deg' not in src:
                continue
            err = great_circle_deg(
                src['azimuth_deg'], src['elevation_deg'],
                src['planned_azimuth_deg'], src['planned_elevation_deg'],
            )
            errors_by_room[room].append(err)

    # ── bottom: DOA snap error by HRTF set ──
    errors_by_hrtf = {}
    for m in metas:
        hrtf = m['hrtf_set']
        if hrtf not in errors_by_hrtf:
            errors_by_hrtf[hrtf] = []
        for src in m['sources']:
            if 'planned_azimuth_deg' not in src:
                continue
            err = great_circle_deg(
                src['azimuth_deg'], src['elevation_deg'],
                src['planned_azimuth_deg'], src['planned_elevation_deg'],
            )
            errors_by_hrtf[hrtf].append(err)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9))

    # ── Top: DOA error by room (violin + box) ──
    ax = axes[0]
    room_names = sorted(errors_by_room.keys())
    room_data = [np.array(errors_by_room[r]) for r in room_names]

    vp = ax.violinplot(room_data, positions=range(len(room_names)),
                       showmedians=False, showextrema=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(ROOM_COLORS.get(room_names[i], '#888'))
        body.set_alpha(0.35)

    bp = ax.boxplot(room_data, positions=range(len(room_names)),
                    widths=0.25, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(ROOM_COLORS.get(room_names[i], '#888'))
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('black')

    ax.set_xticks(range(len(room_names)))
    ax.set_xticklabels([f"{r}\n(n={len(errors_by_room[r])})" for r in room_names])
    ax.set_ylabel("Angular Error (\u00b0)")
    ax.set_title("DOA Snap Error by Room", fontweight='bold')

    # ── Bottom: DOA error by HRTF ──
    ax = axes[1]
    hrtf_names = sorted(errors_by_hrtf.keys())
    hrtf_data = [np.array(errors_by_hrtf[h]) for h in hrtf_names]
    hrtf_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    vp2 = ax.violinplot(hrtf_data, positions=range(len(hrtf_names)),
                        showmedians=False, showextrema=False)
    for i, body in enumerate(vp2['bodies']):
        body.set_facecolor(hrtf_colors[i % len(hrtf_colors)])
        body.set_alpha(0.35)

    bp2 = ax.boxplot(hrtf_data, positions=range(len(hrtf_names)),
                     widths=0.25, patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(hrtf_colors[i % len(hrtf_colors)])
        patch.set_alpha(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp2[element]:
            line.set_color('black')

    ax.set_xticks(range(len(hrtf_names)))
    ax.set_xticklabels([f"{h}\n(n={len(errors_by_hrtf[h])})" for h in hrtf_names])
    ax.set_ylabel("Angular Error (\u00b0)")
    ax.set_title("DOA Snap Error by HRTF Set", fontweight='bold')

    fig.suptitle("Per-Room and Per-HRTF Analysis", fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_room_hrtf_analysis")


# ═══════════════════════════════════════════════════════════════════════════
def _save(fig, name):
    for ext in ['png', 'pdf']:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / name}.{{png,pdf}}")


# ═══════════════════════════════════════════════════════════════════════════
def main():
    plan = load_scene_plan()
    qc   = load_qc()
    with open(SPLITS) as f:
        splits = json.load(f)

    metas = load_meta_batch()

    figure1(plan, qc, splits)
    figure2(metas)
    figure3(qc)
    figure4(metas, qc, plan)

    print("\n=== All figures saved to", FIG_DIR, "===")


if __name__ == "__main__":
    main()
