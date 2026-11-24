"""
High-throughput image corruption detector.
Performs header validation, decompression testing, truncated image checks, and dimension assertions.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageFile
import cv2

# Disallow truncated images to strictly catch corrupted network transfers
ImageFile.LOAD_TRUNCATED_IMAGES = False
logger = logging.getLogger(__name__)

# Valid image magic bytes
MAGIC_NUMBERS = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"RIFF": "WEBP",  # starts with RIFF, bytes 8..12 are WEBP
    b"GIF87a": "GIF",
    b"GIF89a": "GIF",
}


class ImageCorruptionDetector:
    """
    Validates image file integrity across multiple stages:
    1. Header magic byte check
    2. PIL structural integrity (verify)
    3. PIL decompression and decode (load)
    4. OpenCV color decoding & dimension sanity
    """

    def __init__(self, min_width: int = 32, min_height: int = 32):
        self.min_width = min_width
        self.min_height = min_height

    def check_magic_bytes(self, path: Path) -> Tuple[bool, str]:
        """Checks if file starts with valid image magic bytes."""
        try:
            with open(path, "rb") as f:
                header = f.read(16)
            if not header or len(header) < 4:
                return False, "File is empty or too small (<16 bytes)"
            if header.startswith(b"\xff\xd8\xff"):
                return True, "JPEG"
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return True, "PNG"
            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                return True, "WEBP"
            return False, f"Unknown header magic bytes: {header[:8].hex()}"
        except Exception as e:
            return False, f"File read error: {e}"

    def inspect_image(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Runs comprehensive corruption and validity check on an image.
        Returns a detailed inspection dict.
        """
        p = Path(file_path)
        if not p.is_file():
            return {
                "is_valid": False,
                "error_reason": "File does not exist or is a directory",
                "file_path": str(p),
            }

        file_size = p.stat().st_size
        if file_size == 0:
            return {
                "is_valid": False,
                "error_reason": "Zero-byte file",
                "file_size": 0,
                "file_path": str(p),
            }

        # 1. Magic byte check
        magic_ok, fmt_name = self.check_magic_bytes(p)
        if not magic_ok:
            return {
                "is_valid": False,
                "error_reason": f"Corrupt header: {fmt_name}",
                "file_size": file_size,
                "file_path": str(p),
            }

        # 2. PIL verify & load
        try:
            with Image.open(p) as img:
                img.verify()
            # Must re-open to load after verify()
            with Image.open(p) as img:
                img.load()
                width, height = img.size
                mode = img.mode
                channels = len(img.getbands())
        except Exception as e:
            return {
                "is_valid": False,
                "error_reason": f"PIL decompression error: {e}",
                "file_size": file_size,
                "file_path": str(p),
            }

        # 3. Dimension checks
        if width < self.min_width or height < self.min_height:
            return {
                "is_valid": False,
                "error_reason": f"Dimensions too small: {width}x{height} (min: {self.min_width}x{self.min_height})",
                "width": width,
                "height": height,
                "file_size": file_size,
                "file_path": str(p),
            }

        aspect_ratio = width / max(height, 1)
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            return {
                "is_valid": False,
                "error_reason": f"Abnormal aspect ratio: {aspect_ratio:.2f}",
                "width": width,
                "height": height,
                "file_size": file_size,
                "file_path": str(p),
            }

        # 4. OpenCV decoding check
        try:
            cv_img = cv2.imread(str(p))
            if cv_img is None:
                return {
                    "is_valid": False,
                    "error_reason": "OpenCV cv2.imread returned None (corrupt pixel data)",
                    "file_size": file_size,
                    "file_path": str(p),
                }
        except Exception as e:
            return {
                "is_valid": False,
                "error_reason": f"OpenCV read exception: {e}",
                "file_size": file_size,
                "file_path": str(p),
            }

        return {
            "is_valid": True,
            "error_reason": None,
            "width": width,
            "height": height,
            "channels": channels,
            "mode": mode,
            "format": fmt_name,
            "aspect_ratio": round(aspect_ratio, 3),
            "file_size_bytes": file_size,
            "file_path": str(p),
        }
