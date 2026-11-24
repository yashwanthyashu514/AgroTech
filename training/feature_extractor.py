"""
Production-grade Computer Vision Feature Extractor for Agricultural Images.
Extracts color histograms (HSV), Lab moments, domain-specific color ratios,
morphological contour metrics (fruit count, aspect ratio elongation, leaf shape),
and spatial gradient textures.
Runs locally on CPU in < 15ms per image.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    # HSV Histogram bins
    [f"h_bin_{i}" for i in range(16)] +
    [f"s_bin_{i}" for i in range(8)] +
    [f"v_bin_{i}" for i in range(8)] +
    # Lab Color Moments
    ["l_mean", "l_std", "a_mean", "a_std", "a_skew", "b_mean", "b_std", "b_skew"] +
    # RGB Means & Derived
    ["mean_r", "mean_g", "mean_b", "green_excess", "yellow_index"] +
    # Agricultural Color Fractions
    ["red_ratio", "green_ratio", "yellow_ratio", "white_ratio", "brown_ratio", "dark_ratio"] +
    # Morphological Fruit & Leaf Contours
    ["fruit_count", "fruit_avg_elongation", "fruit_area_share", "leaf_avg_elongation"] +
    # Texture & Spatial Gradients
    ["laplacian_var", "sobel_h_energy", "sobel_v_energy", "grad_ratio"]
)


def load_bgr_image(image_input: Union[bytes, str, np.ndarray]) -> Optional[np.ndarray]:
    """Loads image into standard BGR uint8 numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input

    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            try:
                bgr = cv2.imread(image_input)
                if bgr is not None:
                    return bgr
            except Exception:
                pass
            try:
                with open(image_input, "rb") as f:
                    image_input = f.read()
            except Exception:
                return None

        # Decode from bytes
        if CV2_AVAILABLE:
            np_arr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return bgr

        if PIL_AVAILABLE:
            try:
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
                rgb = np.array(img, dtype=np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else rgb
            except Exception:
                pass

    return None


def extract_agri_features(image_input: Union[bytes, str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts high-discriminative agricultural features from an image.
    Returns:
        (feature_vector_1d, metadata_dict)
    """
    bgr = load_bgr_image(image_input)
    if bgr is None:
        # Default zeros
        zeros = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        return zeros, {"status": "error", "message": "Failed to decode image"}

    # Resize to standard analysis resolution (max 400px)
    h, w = bgr.shape[:2]
    scale = min(400.0 / max(h, w), 1.0)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    total_pixels = float(bgr.shape[0] * bgr.shape[1])

    # 1. RGB Metrics
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    green_excess = float(mean_g - (mean_r + mean_b) / 2.0)
    yellow_index = float((mean_r + mean_g) / max(2.0 * mean_b + 1e-5, 1.0))

    # 2. HSV Color Space
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Histograms
    h_hist, _ = np.histogram(h_channel, bins=16, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s_channel, bins=8, range=(0, 256), density=True)
    v_hist, _ = np.histogram(v_channel, bins=8, range=(0, 256), density=True)

    # 3. Lab Color Space & Moments
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_c = lab[:, :, 0].astype(np.float32)
    a_c = lab[:, :, 1].astype(np.float32) - 128.0
    b_c = lab[:, :, 2].astype(np.float32) - 128.0

    l_mean, l_std = float(np.mean(l_c)), float(np.std(l_c))
    a_mean, a_std = float(np.mean(a_c)), float(np.std(a_c))
    a_skew = float(np.mean(((a_c - a_mean) / (a_std + 1e-5)) ** 3))
    b_mean, b_std = float(np.mean(b_c)), float(np.std(b_c))
    b_skew = float(np.mean(((b_c - b_mean) / (b_std + 1e-5)) ** 3))

    # 4. Agricultural Color Fractions
    # Red mask (Ripe chillies, ripe tomatoes): H in [0, 14] or [166, 180], S > 60, V > 45
    m_red1 = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([14, 255, 255]))
    m_red2 = cv2.inRange(hsv, np.array([166, 60, 45]), np.array([180, 255, 255]))
    mask_red = m_red1 | m_red2
    red_ratio = float(np.sum(mask_red > 0) / total_pixels)

    # Green foliage mask: H in [28, 88], S > 35, V > 35
    mask_green = cv2.inRange(hsv, np.array([28, 35, 35]), np.array([88, 255, 255]))
    green_ratio = float(np.sum(mask_green > 0) / total_pixels)

    # Yellow chlorosis / panicles: H in [16, 28], S > 50, V > 60
    mask_yellow = cv2.inRange(hsv, np.array([16, 50, 60]), np.array([28, 255, 255]))
    yellow_ratio = float(np.sum(mask_yellow > 0) / total_pixels)

    # White cotton lint / powdery mildew: S < 40, V > 180
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 45, 255]))
    white_ratio = float(np.sum(mask_white > 0) / total_pixels)

    # Brown soil / necrotic tissue: R in [80, 180], G in [50, 140], B < 100, R > B + 25
    mask_brown = (r > 80) & (r < 190) & (g > 45) & (g < 150) & (b < 110) & (r > b + 20)
    brown_ratio = float(np.sum(mask_brown) / total_pixels)

    # Deep dark / necrotic spots: r, g, b < 65
    mask_dark = (r < 65) & (g < 65) & (b < 65)
    dark_ratio = float(np.sum(mask_dark) / total_pixels)

    # 5. Morphological Fruit & Pod Contours
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruit_elongations = []
    total_fruit_area = 0.0
    for c in contours_red:
        area = cv2.contourArea(c)
        if area > 25:
            total_fruit_area += area
            _, _, cw, ch = cv2.boundingRect(c)
            elongation = max(cw, ch) / float(min(cw, ch) + 1e-5)
            fruit_elongations.append(elongation)

    fruit_count = len(fruit_elongations)
    fruit_avg_elongation = float(np.mean(fruit_elongations)) if fruit_elongations else 1.0
    fruit_area_share = float(total_fruit_area / total_pixels)

    # 6. Morphological Leaf Contours
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_elongations = []
    for c in contours_green:
        if cv2.contourArea(c) > 40:
            _, _, lw, lh = cv2.boundingRect(c)
            elongation = max(lw, lh) / float(min(lw, lh) + 1e-5)
            leaf_elongations.append(elongation)

    leaf_avg_elongation = float(np.mean(leaf_elongations)) if leaf_elongations else 1.0

    # 7. Texture & Gradients (Gray channel)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_h_energy = float(np.mean(np.abs(sobel_h)))
    sobel_v_energy = float(np.mean(np.abs(sobel_v)))
    grad_ratio = float((sobel_h_energy + 1e-5) / (sobel_v_energy + 1e-5))

    # Assemble vector in exact feature order
    feat_dict = {
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "green_excess": green_excess, "yellow_index": yellow_index,
        "l_mean": l_mean, "l_std": l_std,
        "a_mean": a_mean, "a_std": a_std, "a_skew": a_skew,
        "b_mean": b_mean, "b_std": b_std, "b_skew": b_skew,
        "red_ratio": red_ratio, "green_ratio": green_ratio,
        "yellow_ratio": yellow_ratio, "white_ratio": white_ratio,
        "brown_ratio": brown_ratio, "dark_ratio": dark_ratio,
        "fruit_count": fruit_count,
        "fruit_avg_elongation": fruit_avg_elongation,
        "fruit_area_share": fruit_area_share,
        "leaf_avg_elongation": leaf_avg_elongation,
        "laplacian_var": laplacian_var,
        "sobel_h_energy": sobel_h_energy,
        "sobel_v_energy": sobel_v_energy,
        "grad_ratio": grad_ratio,
    }

    values = []
    # 1. H hist
    values.extend(h_hist)
    # 2. S hist
    values.extend(s_hist)
    # 3. V hist
    values.extend(v_hist)
    # 4. Lab
    values.extend([l_mean, l_std, a_mean, a_std, a_skew, b_mean, b_std, b_skew])
    # 5. RGB
    values.extend([mean_r, mean_g, mean_b, green_excess, yellow_index])
    # 6. Colors
    values.extend([red_ratio, green_ratio, yellow_ratio, white_ratio, brown_ratio, dark_ratio])
    # 7. Contours
    values.extend([fruit_count, fruit_avg_elongation, fruit_area_share, leaf_avg_elongation])
    # 8. Gradients
    values.extend([laplacian_var, sobel_h_energy, sobel_v_energy, grad_ratio])

    feature_vec = np.array(values, dtype=np.float32)

    # Boolean flags for semantic routing
    has_fruits = (red_ratio > 0.035 or fruit_count >= 2)
    is_healthy_fruit = (has_fruits and green_ratio > 0.20 and dark_ratio < 0.20)
    is_soil = (green_excess < -5.0 and (brown_ratio > 0.25 or dark_ratio > 0.3) and not has_fruits)
    is_pest = (dark_ratio > 0.20 and green_excess > 5.0 and not has_fruits)

    meta = {
        "has_fruits": has_fruits,
        "is_healthy_fruit": is_healthy_fruit,
        "is_soil": is_soil,
        "is_pest": is_pest,
        "fruit_count": fruit_count,
        "fruit_avg_elongation": fruit_avg_elongation,
        "red_ratio": red_ratio,
        "green_ratio": green_ratio,
        "dark_ratio": dark_ratio,
        "leaf_avg_elongation": leaf_avg_elongation,
    }

    return feature_vec, meta
