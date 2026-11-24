"""
Safe archive extractor with path traversal (Zip Slip) protection and extraction provenance.
"""
from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SafeArchiveExtractor:
    """
    Safely unpacks ZIP or TAR archives while preventing directory traversal (Zip Slip).
    """

    def __init__(self, extracted_dir: str = "data_pipeline/extracted"):
        self.extracted_dir = Path(extracted_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    def _is_safe_path(self, target_base: Path, member_path: str) -> bool:
        """Ensures the member cannot escape target_base directory."""
        resolved_target = (target_base / member_path).resolve()
        return str(resolved_target).startswith(str(target_base.resolve()))

    def extract_zip(self, archive_path: Path, output_dir: Path) -> List[str]:
        """Safely extracts a ZIP file."""
        extracted_files: List[str] = []
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                if not self._is_safe_path(output_dir, member.filename):
                    logger.warning("Unsafe path detected in zip: %s, skipping!", member.filename)
                    continue
                target_file = output_dir / member.filename
                target_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(target_file, "wb") as target:
                    target.write(source.read())
                extracted_files.append(str(target_file))
        return extracted_files

    def extract_tar(self, archive_path: Path, output_dir: Path) -> List[str]:
        """Safely extracts a TAR archive."""
        extracted_files: List[str] = []
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                if not self._is_safe_path(output_dir, member.name):
                    logger.warning("Unsafe path detected in tar: %s, skipping!", member.name)
                    continue
                tf.extract(member, path=output_dir)
                extracted_files.append(str(output_dir / member.name))
        return extracted_files

    def extract(self, archive_path: str | Path, dataset_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Unpacks the given archive into data_pipeline/extracted/<dataset_id>/.
        """
        arch = Path(archive_path)
        if not arch.exists():
            raise FileNotFoundError(f"Archive not found: {arch}")

        target_dir = self.extracted_dir / dataset_id
        if target_dir.exists() and any(target_dir.iterdir()) and not force:
            logger.info("Dataset '%s' already extracted at %s", dataset_id, target_dir)
            files = [str(p) for p in target_dir.rglob("*") if p.is_file()]
            return {
                "status": "cached",
                "extracted_dir": str(target_dir),
                "total_files": len(files),
                "dataset_id": dataset_id,
            }

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s to %s ...", arch.name, target_dir)

        if zipfile.is_zipfile(arch):
            extracted = self.extract_zip(arch, target_dir)
        elif tarfile.is_tarfile(arch):
            extracted = self.extract_tar(arch, target_dir)
        else:
            raise ValueError(f"Unsupported archive format: {arch.name}")

        logger.info("Successfully extracted %d files for dataset %s", len(extracted), dataset_id)
        return {
            "status": "extracted",
            "extracted_dir": str(target_dir),
            "total_files": len(extracted),
            "dataset_id": dataset_id,
        }
