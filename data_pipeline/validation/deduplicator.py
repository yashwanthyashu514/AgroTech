"""
Exact and Perceptual Deduplication Engine.
Implements SHA-256 for identical file matching and 64-bit dHash (difference hashing)
with Hamming distance graph clustering for perceptual near-duplicate detection.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PerceptualDeduplicator:
    """
    Computes exact SHA256 and perceptual dHash for images.
    Clusters near-duplicates to prevent data leakage during train/val/test splitting.
    """

    def __init__(self, hamming_threshold: int = 4):
        """
        Args:
            hamming_threshold: Maximum bit differences (out of 64) to consider two images as near-duplicates.
                              Standard: <=4 detects crops, brightness shifts, slight rotations, or video burst frames.
        """
        self.hamming_threshold = hamming_threshold

    @staticmethod
    def compute_sha256(file_path: str | Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_dhash(file_path: str | Path) -> str:
        """
        Computes 64-bit difference hash (dHash).
        1. Convert to grayscale
        2. Resize to 9x8 (width=9, height=8)
        3. Compare adjacent pixels in each row
        """
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "0000000000000000"
        # Resize to 9 cols by 8 rows
        resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        # Compute difference: row col > row col+1
        diff = resized[:, 1:] > resized[:, :-1]
        # Convert 8x8 boolean matrix to 64-bit integer
        decimal_val = 0
        for bit in diff.flatten():
            decimal_val = (decimal_val << 1) | int(bit)
        return f"{decimal_val:016x}"

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Calculates Hamming distance between two 16-character hex hashes (64 bits)."""
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count("1")

    def cluster_duplicates(
        self, records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Takes a list of records with 'file_path', 'sha256', and 'dhash'.
        Assigns a 'duplicate_cluster_id' to each record.
        Images with identical SHA256 or dHash distance <= threshold share the same cluster_id.
        """
        n = len(records)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # 1. Exact duplicates grouping via SHA-256
        sha_map: Dict[str, int] = {}
        exact_duplicate_count = 0
        for i, rec in enumerate(records):
            sha = rec.get("sha256", "")
            if sha in sha_map:
                union(i, sha_map[sha])
                exact_duplicate_count += 1
            else:
                sha_map[sha] = i

        # 2. Perceptual near-duplicates grouping via dHash
        # Optimization: group by crop or compare within candidate blocks
        near_duplicate_pairs = 0
        for i in range(n):
            hash_i = records[i].get("dhash", "")
            if not hash_i or hash_i == "0000000000000000":
                continue
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                hash_j = records[j].get("dhash", "")
                if not hash_j or hash_j == "0000000000000000":
                    continue
                dist = self.hamming_distance(hash_i, hash_j)
                if dist <= self.hamming_threshold:
                    union(i, j)
                    near_duplicate_pairs += 1

        # 3. Assign canonical cluster IDs
        cluster_map: Dict[int, str] = {}
        cluster_sizes: Dict[str, int] = {}
        cluster_counter = 1

        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cid = f"cluster_{cluster_counter:06d}"
                cluster_map[root] = cid
                cluster_counter += 1
            else:
                cid = cluster_map[root]

            records[i]["duplicate_cluster_id"] = cid
            records[i]["is_duplicate"] = (root != i) or (records[i].get("sha256") in sha_map and sha_map[records[i]["sha256"]] != i)
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        multi_item_clusters = sum(1 for sz in cluster_sizes.values() if sz > 1)
        total_duplicates_found = sum(sz - 1 for sz in cluster_sizes.values() if sz > 1)

        stats = {
            "total_items": n,
            "unique_clusters": len(cluster_sizes),
            "multi_item_clusters": multi_item_clusters,
            "exact_duplicate_count": exact_duplicate_count,
            "near_duplicate_pairs": near_duplicate_pairs,
            "total_redundant_images": total_duplicates_found,
        }

        return records, stats
