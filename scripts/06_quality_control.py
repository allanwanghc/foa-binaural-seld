#!/usr/bin/env python3
"""
Step 6: Quality control checks on generated scenes.

Runs QC checks on all scenes and produces a report CSV.
"""

import sys
import csv
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SCENES_DIR, OUTPUT_DIR, TARGET_SR
from src.qc import run_qc_scene


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run QC on generated scenes")
    parser.add_argument("--test", action="store_true", help="Test mode: 10 scenes")
    args = parser.parse_args()

    scene_dirs = sorted(SCENES_DIR.glob("scene_*"))
    if args.test:
        scene_dirs = scene_dirs[:10]

    print(f"{'=' * 60}")
    print(f"Quality Control")
    print(f"  Scenes to check: {len(scene_dirs)}")
    print(f"{'=' * 60}\n")

    report_path = OUTPUT_DIR / "qc_report.csv"
    all_results = []
    total_passed = 0
    total_failed = 0

    for scene_dir in tqdm(scene_dirs, desc="QC checks"):
        results = run_qc_scene(scene_dir, TARGET_SR)
        passed = all(r[0] for r in results.values())

        row = {"scene": scene_dir.name, "passed": passed}
        for check_name, (ok, detail) in results.items():
            row[check_name] = "PASS" if ok else f"FAIL: {detail}"

        all_results.append(row)
        if passed:
            total_passed += 1
        else:
            total_failed += 1

    # Save report
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n{'=' * 60}")
    print(f"QC Report: {report_path}")
    print(f"  Total: {len(all_results)}")
    print(f"  Passed: {total_passed} ({100*total_passed/max(1,len(all_results)):.1f}%)")
    print(f"  Failed: {total_failed}")

    if total_failed > 0:
        print(f"\n  Failed scenes:")
        for row in all_results:
            if not row["passed"]:
                fails = [k for k, v in row.items() if isinstance(v, str) and v.startswith("FAIL")]
                print(f"    {row['scene']}: {', '.join(fails)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
