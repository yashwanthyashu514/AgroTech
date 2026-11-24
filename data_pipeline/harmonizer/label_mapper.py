"""
Harmonizes varied class naming schemes across PlantVillage, PlantDoc, IP102,
and regional Karnataka field imagery into a canonical taxonomy.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LabelHarmonizer:
    """
    Standardizes crop and disease/pest labels into unified schema:
    - canonical_crop: lowercase standard crop name (e.g., 'rice', 'tomato', 'ragi')
    - canonical_condition: standardized disease or pest key (e.g., 'early_blight', 'healthy')
    - condition_type: 'disease' | 'pest' | 'deficiency' | 'healthy' | 'soil'
    - canonical_label: '{canonical_crop}___{canonical_condition}'
    """

    def __init__(self, mapping_path: Optional[str] = None, crops_path: Optional[str] = None):
        if mapping_path is None:
            mapping_path = str(Path(__file__).parent.parent / "configs" / "label_mapping.json")
        if crops_path is None:
            crops_path = str(Path(__file__).parent.parent / "configs" / "karnataka_crops.json")

        with open(mapping_path, "r", encoding="utf-8") as f:
            self.known_mappings: Dict[str, Dict[str, Any]] = json.load(f).get("mappings", {})

        with open(crops_path, "r", encoding="utf-8") as f:
            self.karnataka_crops = json.load(f).get("crops", {})

        self.crop_aliases = {
            "paddy": "rice",
            "corn": "maize",
            "finger_millet": "ragi",
            "fingermillet": "ragi",
            "betelnut": "arecanut",
            "betel_nut": "arecanut",
            "cotton": "cotton",
            "sugarcane": "sugarcane",
            "coconut": "coconut",
            "wheat": "wheat",
            "tomato": "tomato",
        }

    def harmonize(self, raw_label: str) -> Dict[str, str]:
        """
        Takes a raw label (e.g. folder name or class tag) and maps it to canonical fields.
        """
        clean_raw = raw_label.strip()

        # 1. Exact match in predefined mapping
        if clean_raw in self.known_mappings:
            entry = self.known_mappings[clean_raw]
            return {
                "raw_label": clean_raw,
                "canonical_crop": entry["canonical_crop"],
                "canonical_condition": entry["canonical_disease"],
                "condition_type": entry["condition_type"],
                "status": entry["status"],
                "canonical_label": entry["canonical_label"],
            }

        # 2. PlantVillage style: "Crop___Condition"
        if "___" in clean_raw:
            parts = clean_raw.split("___", 1)
            raw_crop, raw_cond = parts[0], parts[1]
        elif "_" in clean_raw and not clean_raw.lower().startswith("cluster"):
            # Try splitting first word as crop
            parts = clean_raw.split("_", 1)
            raw_crop, raw_cond = parts[0], parts[1]
        else:
            raw_crop, raw_cond = clean_raw, "unspecified"

        # Normalize crop name
        crop_clean = re.sub(r"[^a-zA-Z]", "", raw_crop).lower()
        canonical_crop = self.crop_aliases.get(crop_clean, crop_clean)
        if canonical_crop not in self.karnataka_crops:
            canonical_crop = "all_other_crops"

        # Normalize condition
        cond_clean = re.sub(r"[^a-zA-Z0-9_]", "_", raw_cond).lower().strip("_")
        if "healthy" in cond_clean:
            condition_type = "healthy"
            canonical_condition = "healthy"
            status = "healthy"
        elif any(p in cond_clean for p in ["borer", "armyworm", "mite", "aphid", "beetle", "caterpillar", "bug"]):
            condition_type = "pest"
            canonical_condition = cond_clean
            status = "diseased"
        elif any(d in cond_clean for d in ["nitrogen", "potassium", "phosphorus", "deficiency", "chlorosis"]):
            condition_type = "deficiency"
            canonical_condition = cond_clean
            status = "diseased"
        else:
            condition_type = "disease"
            canonical_condition = cond_clean
            status = "diseased"

        canonical_label = f"{canonical_crop}___{canonical_condition}"

        return {
            "raw_label": clean_raw,
            "canonical_crop": canonical_crop,
            "canonical_condition": canonical_condition,
            "condition_type": condition_type,
            "status": status,
            "canonical_label": canonical_label,
        }
