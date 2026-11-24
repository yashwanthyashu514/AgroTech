"""
Stage 2 CLI Runner — Automated Dataset Acquisition, Validation, and Integration.
Orchestrates downloading, safe extraction, corruption detection, perceptual deduplication,
label harmonization, leakage-free splitting, and statistical reporting.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.acquisition.downloader import DatasetDownloader
from data_pipeline.acquisition.extractor import SafeArchiveExtractor
from data_pipeline.manifest.builder import ManifestBuilder
from data_pipeline.reporting.stats_reporter import DatasetQualityReporter
from data_pipeline.splitter.leakage_free_split import LeakageFreeSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("Stage2Pipeline")


def generate_synthetic_karnataka_testbed(target_dir: Path) -> Path:
    """
    Generates realistic synthetic test images for Karnataka's 9 priority crops
    (Rice, Tomato, Maize, Cotton, Sugarcane, Ragi, Arecanut, Coconut, Wheat)
    including healthy and diseased variants, intentional near-duplicates,
    and an intentionally corrupt file to test the filter.
    """
    import cv2
    import numpy as np

    logger.info("Generating synthetic verification testbed at %s ...", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    crops_conditions = [
        ("rice", ["blast", "bacterial_blight", "brown_spot", "healthy"]),
        ("tomato", ["early_blight", "late_blight", "leaf_curl", "bacterial_spot", "healthy"]),
        ("maize", ["common_rust", "northern_leaf_blight", "fall_armyworm", "healthy"]),
        ("cotton", ["bacterial_blight", "pink_bollworm", "healthy"]),
        ("sugarcane", ["red_rot", "smut", "healthy"]),
        ("ragi", ["blast", "brown_spot", "healthy"]),
        ("arecanut", ["koleroga", "yellow_leaf", "healthy"]),
        ("coconut", ["bud_rot", "stem_bleeding", "healthy"]),
        ("wheat", ["stripe_rust", "leaf_rust", "healthy"]),
    ]

    np.random.seed(42)
    sample_count = 0

    for crop, conditions in crops_conditions:
        for cond in conditions:
            folder_name = f"{crop.capitalize()}___{cond}"
            class_dir = target_dir / folder_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Generate 3 base images per condition
            for idx in range(1, 4):
                # Create a 256x256 image with characteristic agricultural colors
                img = np.zeros((256, 256, 3), dtype=np.uint8)
                # Seed per crop and condition to create distinct visual clusters
                c_seed = abs(hash(f"{crop}_{cond}_{idx}")) % 1000
                rng = np.random.RandomState(c_seed)
                base_color = rng.randint(40, 220, size=3)
                img[:, :] = base_color

                # Add distinct simulated shapes
                center = (rng.randint(50, 200), rng.randint(50, 200))
                radius = rng.randint(20, 50)
                color = (int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255)))
                cv2.circle(img, center, radius, color, -1)
                pt1 = (rng.randint(0, 100), rng.randint(0, 100))
                pt2 = (rng.randint(150, 255), rng.randint(150, 255))
                cv2.line(img, pt1, pt2, (int(rng.randint(0, 255)), int(rng.randint(0, 255)), int(rng.randint(0, 255))), 3)

                img_path = class_dir / f"{crop}_{cond}_{idx:03d}.jpg"
                cv2.imwrite(str(img_path), img)
                sample_count += 1

                # Generate a near-duplicate for idx 1 (subtle brightness shift) to test dHash clustering
                if idx == 1:
                    near_dup = np.clip(img.astype(np.int16) + 3, 0, 255).astype(np.uint8)
                    dup_path = class_dir / f"{crop}_{cond}_{idx:03d}_burst_frame2.jpg"
                    cv2.imwrite(str(dup_path), near_dup)
                    sample_count += 1

    # Add an intentionally corrupted file to verify detection
    corrupt_dir = target_dir / "Tomato___corrupted_sample"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = corrupt_dir / "corrupted_zero_byte.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"")  # 0 bytes

    corrupt_file2 = corrupt_dir / "corrupted_bad_header.jpg"
    with open(corrupt_file2, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER_RANDOM_GARBAGE_BYTES")

    logger.info("Generated %d synthetic sample images (including 9 duplicates and 2 corrupt files).", sample_count)
    return target_dir


def run_pipeline(
    sample_run: bool = False,
    source_dataset: Optional[str] = None,
    custom_scan_dir: Optional[str] = None,
    hamming_threshold: int = 4,
    output_dir: str = "data_pipeline/manifests",
    reports_dir: str = "data_pipeline/reports",
) -> Dict[str, Any]:
    """
    Executes the entire Stage 2 pipeline.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("🚀 KARNATAKA AGRICULTURE AI AGENT — STAGE 2 PIPELINE")
    logger.info("=" * 70)

    downloader = DatasetDownloader()
    extractor = SafeArchiveExtractor()
    builder = ManifestBuilder(output_dir=output_dir, hamming_threshold=hamming_threshold)
    splitter = LeakageFreeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    reporter = DatasetQualityReporter(reports_dir=reports_dir)

    all_valid: List[Dict[str, Any]] = []
    all_corrupt: List[Dict[str, Any]] = []

    # ── Step 1: Dataset Acquisition / Scan ────────────────────────────────────
    if sample_run:
        logger.info("\n[Step 1/5] Running with Karnataka verification sample testbed ...")
        sample_path = Path("data_pipeline/sample_testbed")
        generate_synthetic_karnataka_testbed(sample_path)
        valid, corrupt = builder.scan_dataset_directory(
            root_dir=sample_path,
            source_dataset_id="karnataka_synthetic_verification",
            provenance_license="Internal Verification",
        )
        all_valid.extend(valid)
        all_corrupt.extend(corrupt)
    elif custom_scan_dir:
        logger.info("\n[Step 1/5] Scanning custom input directory: %s ...", custom_scan_dir)
        valid, corrupt = builder.scan_dataset_directory(
            root_dir=custom_scan_dir,
            source_dataset_id="local_custom_scan",
        )
        all_valid.extend(valid)
        all_corrupt.extend(corrupt)
    elif source_dataset:
        logger.info("\n[Step 1/5] Downloading and acquiring '%s' ...", source_dataset)
        dl_res = downloader.download_dataset(source_dataset)
        if dl_res.get("status") in ["downloaded", "cached"]:
            ext_res = extractor.extract(dl_res["archive_path"], dataset_id=source_dataset)
            valid, corrupt = builder.scan_dataset_directory(
                root_dir=ext_res["extracted_dir"],
                source_dataset_id=source_dataset,
            )
            all_valid.extend(valid)
            all_corrupt.extend(corrupt)
    else:
        logger.info("No source specified. Scanning data_pipeline/extracted/ ...")
        ext_root = Path("data_pipeline/extracted")
        if ext_root.exists():
            for d in ext_root.iterdir():
                if d.is_dir():
                    valid, corrupt = builder.scan_dataset_directory(d, source_dataset_id=d.name)
                    all_valid.extend(valid)
                    all_corrupt.extend(corrupt)

    logger.info(
        "\n[Step 2/5] Validation & Deduplication Complete: %d valid images, %d corrupt filtered.",
        len(all_valid), len(all_corrupt)
    )

    if not all_valid:
        logger.warning("No valid images found to process!")
        return {"status": "no_data"}

    # ── Step 3: Leakage-Free Stratified Splitting ─────────────────────────────
    logger.info("\n[Step 3/5] Applying leakage-free cluster-isolated splitting ...")
    all_valid, split_summary = splitter.split(all_valid)

    # ── Step 4: Manifest Generation ───────────────────────────────────────────
    logger.info("\n[Step 4/5] Generating unified CSV and JSON manifests ...")
    manifest_paths = builder.save_manifests(all_valid, all_corrupt)

    # Re-run deduplicator stats aggregation for reporter
    all_valid, dedup_stats = builder.deduplicator.cluster_duplicates(all_valid)

    # ── Step 5: Quality & Statistics Reporting ────────────────────────────────
    logger.info("\n[Step 5/5] Generating comprehensive dataset quality reports ...")
    report_paths = reporter.generate_report(
        valid_records=all_valid,
        corrupted_records=all_corrupt,
        split_summary=split_summary,
        dedup_stats=dedup_stats,
    )

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("✅ STAGE 2 PIPELINE COMPLETED IN %.2f SECONDS", elapsed)
    logger.info("   Manifest CSV:    %s", manifest_paths["csv_path"])
    logger.info("   Statistics MD:   %s", report_paths["markdown_path"])
    logger.info("=" * 70)

    return {
        "status": "success",
        "valid_count": len(all_valid),
        "corrupt_count": len(all_corrupt),
        "manifest_paths": manifest_paths,
        "report_paths": report_paths,
        "split_summary": split_summary,
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karnataka Agriculture AI Agent - Stage 2 Pipeline")
    parser.add_argument("--sample-run", action="store_true", help="Run with synthetic Karnataka testbed to verify pipeline")
    parser.add_argument("--source", type=str, help="Dataset ID to download and process from registry")
    parser.add_argument("--scan-dir", type=str, help="Local directory of images to process")
    parser.add_argument("--hamming-threshold", type=int, default=4, help="Hamming distance threshold for dHash deduplication")

    args = parser.parse_args()
    run_pipeline(
        sample_run=args.sample_run,
        source_dataset=args.source,
        custom_scan_dir=args.scan_dir,
        hamming_threshold=args.hamming_threshold,
    )
