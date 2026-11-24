"""
Unified Manifest Builder.
Walks extracted dataset trees, validates image integrity, deduplicates near-identical images,
harmonizes labels, and generates standardized production CSV and JSON manifests.
"""
from __future__ import annotations

import csv
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_pipeline.harmonizer.label_mapper import LabelHarmonizer
from data_pipeline.validation.corrupt_detector import ImageCorruptionDetector
from data_pipeline.validation.deduplicator import PerceptualDeduplicator

logger = logging.getLogger(__name__)


class ManifestBuilder:
    """
    Builds the central unified dataset manifest for the Karnataka Agriculture AI Agent.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        output_dir: str = "data_pipeline/manifests",
        registry_path: Optional[str] = None,
        hamming_threshold: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = ImageCorruptionDetector()
        self.deduplicator = PerceptualDeduplicator(hamming_threshold=hamming_threshold)
        self.harmonizer = LabelHarmonizer()

        if registry_path is None:
            registry_path = str(Path(__file__).parent.parent / "configs" / "dataset_registry.json")
        with open(registry_path, "r", encoding="utf-8") as f:
            self.registry = json.load(f).get("registry", {})

    def scan_dataset_directory(
        self,
        root_dir: str | Path,
        source_dataset_id: str = "unknown_source",
        provenance_license: str = "Unspecified",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scans a directory tree where subdirectories are class names.
        e.g., root_dir/Tomato___Early_blight/image001.jpg
        """
        root = Path(root_dir)
        valid_records: List[Dict[str, Any]] = []
        corrupted_records: List[Dict[str, Any]] = []

        if not root.exists():
            logger.warning("Directory does not exist: %s", root)
            return [], []

        # Lookup license from registry if known
        if source_dataset_id in self.registry:
            provenance_license = self.registry[source_dataset_id].get("license", provenance_license)

        # Collect all image files
        all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS]
        logger.info("Scanning %d images in %s ...", len(all_files), root)

        for p in all_files:
            # 1. Corruption and sanity check
            inspection = self.detector.inspect_image(p)
            if not inspection["is_valid"]:
                corrupted_records.append({
                    "file_path": str(p.relative_to(root.parent)),
                    "error_reason": inspection["error_reason"],
                    "file_size_bytes": inspection.get("file_size", 0),
                    "source_dataset": source_dataset_id,
                })
                continue

            # 2. Extract raw label from parent folder name
            raw_label = p.parent.name
            label_info = self.harmonizer.harmonize(raw_label)

            # 3. Hashes for exact and near-duplicate detection
            sha256 = self.deduplicator.compute_sha256(p)
            dhash = self.deduplicator.compute_dhash(p)

            record = {
                "image_id": str(uuid.uuid4()),
                "file_path": str(p).replace("\\", "/"),
                "relative_path": str(p.relative_to(root)).replace("\\", "/"),
                "file_name": p.name,
                "source_dataset": source_dataset_id,
                "provenance_license": provenance_license,
                "raw_label": label_info["raw_label"],
                "canonical_crop": label_info["canonical_crop"],
                "canonical_condition": label_info["canonical_condition"],
                "condition_type": label_info["condition_type"],
                "status": label_info["status"],
                "canonical_label": label_info["canonical_label"],
                "width": inspection["width"],
                "height": inspection["height"],
                "channels": inspection["channels"],
                "aspect_ratio": inspection["aspect_ratio"],
                "format": inspection["format"],
                "file_size_bytes": inspection["file_size_bytes"],
                "sha256": sha256,
                "dhash": dhash,
            }
            valid_records.append(record)

        # 4. Run clustering on valid records
        if valid_records:
            valid_records, dup_stats = self.deduplicator.cluster_duplicates(valid_records)
            logger.info("Deduplication stats for %s: %s", source_dataset_id, dup_stats)

        return valid_records, corrupted_records

    def save_manifests(
        self,
        valid_records: List[Dict[str, Any]],
        corrupted_records: List[Dict[str, Any]],
        manifest_filename: str = "unified_dataset_manifest.csv",
    ) -> Dict[str, str]:
        """
        Saves CSV and JSON manifests.
        """
        csv_path = self.output_dir / manifest_filename
        json_path = self.output_dir / manifest_filename.replace(".csv", ".json")
        corrupt_csv_path = self.output_dir / "corrupted_files.csv"

        fieldnames = [
            "image_id", "file_path", "relative_path", "source_dataset", "provenance_license",
            "canonical_crop", "canonical_condition", "condition_type", "status",
            "canonical_label", "duplicate_cluster_id", "is_duplicate", "width", "height",
            "channels", "aspect_ratio", "format", "file_size_bytes", "sha256", "dhash",
            "split"
        ]

        # Write valid manifest CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in valid_records:
                if "split" not in r:
                    r["split"] = "unassigned"
                writer.writerow(r)

        # Write valid manifest JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(valid_records, f, indent=2)

        # Write corrupt files CSV
        if corrupted_records:
            corrupt_fields = ["file_path", "error_reason", "file_size_bytes", "source_dataset"]
            with open(corrupt_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=corrupt_fields)
                writer.writeheader()
                writer.writerows(corrupted_records)

        logger.info("Saved %d valid records to %s", len(valid_records), csv_path)
        logger.info("Saved %d corrupt records to %s", len(corrupted_records), corrupt_csv_path)

        return {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "corrupt_csv_path": str(corrupt_csv_path) if corrupted_records else None,
        }
