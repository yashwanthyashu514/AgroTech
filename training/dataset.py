"""
PyTorch Dataset for Agricultural Multimodal Manifests.
Loads unified CSV manifests with in-field data augmentations tailored for Karnataka crop imagery.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class AgricultureDataset:
    """
    Standard PyTorch-compatible dataset for multi-task agricultural classification:
    Head 1: Crop ID
    Head 2: Disease / Condition ID
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str = "train",
        transform: Optional[Any] = None,
        crop_to_idx: Optional[Dict[str, int]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.split = split
        self.transform = transform
        self.records: List[Dict[str, str]] = []

        manifest_path = Path(manifest_csv)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

        # Load records for this split
        all_crops = set()
        all_labels = set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_crops.add(row["canonical_crop"])
                all_labels.add(row["canonical_label"])
                if row.get("split") == split or split == "all":
                    self.records.append(row)

        self.crop_to_idx = crop_to_idx or {c: i for i, c in enumerate(sorted(all_crops))}
        self.label_to_idx = label_to_idx or {l: i for i, l in enumerate(sorted(all_labels))}
        self.idx_to_crop = {v: k for k, v in self.crop_to_idx.items()}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        logger.info(
            "Loaded split='%s': %d samples | %d crops | %d disease classes",
            split, len(self.records), len(self.crop_to_idx), len(self.label_to_idx)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Any, int, int, Dict[str, Any]]:
        row = self.records[idx]
        image_path = Path(row["file_path"])

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            logger.warning("Failed to open image %s: %s", image_path, e)
            image = Image.new("RGB", (256, 256), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        crop_idx = self.crop_to_idx.get(row["canonical_crop"], 0)
        label_idx = self.label_to_idx.get(row["canonical_label"], 0)

        metadata = {
            "image_id": row["image_id"],
            "canonical_crop": row["canonical_crop"],
            "canonical_label": row["canonical_label"],
            "status": row.get("status", "unknown"),
            "file_path": row["file_path"],
        }

        return image, crop_idx, label_idx, metadata
