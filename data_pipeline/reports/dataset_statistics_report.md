# Karnataka Agriculture AI Agent — Dataset Statistics & Quality Report

**Generated:** Automated Stage 2 Pipeline · **Target Region:** Karnataka

## 1. Executive Summary

| Metric | Count | Details |
| :--- | :--- | :--- |
| **Total Images Processed** | `126` | Full raw catalog |
| **Valid Images** | `124` | Header, decompression & OpenCV verified |
| **Corrupted Images Filtered** | `2` | Zero-byte, truncated, or bad magic headers |
| **Exact Duplicates (SHA-256)** | `5` | Identical file bytes detected |
| **Perceptual Near-Duplicates** | `36` | Hamming distance ≤ 4 (dHash) |
| **Isolated Duplicate Clusters** | `83` | Grouped to prevent data leakage |
| **Data Leakage Guarantee** | `0% Leakage` | Train / Val / Test cluster intersection is null |

## 2. Train / Validation / Test Splits

| Split | Samples | Percentage | Duplicate Clusters |
| :--- | :--- | :--- | :--- |
| **Train** | `50` | `40.3%` | `33` |
| **Validation** | `38` | `30.6%` | `25` |
| **Test** | `36` | `29.0%` | `25` |

## 3. Crop Distribution (Karnataka Priority Crops)

| Crop | Images | Share |
| :--- | :--- | :--- |
| **Tomato** | `20` | `16.1%` |
| **Maize** | `16` | `12.9%` |
| **Rice** | `16` | `12.9%` |
| **Arecanut** | `12` | `9.7%` |
| **Coconut** | `12` | `9.7%` |
| **Cotton** | `12` | `9.7%` |
| **Ragi** | `12` | `9.7%` |
| **Sugarcane** | `12` | `9.7%` |
| **Wheat** | `12` | `9.7%` |

## 4. Condition Category Distribution

| Category | Images | Percentage |
| :--- | :--- | :--- |
| **Disease** | `84` | `67.7%` |
| **Healthy** | `36` | `29.0%` |
| **Pest** | `4` | `3.2%` |

## 5. Class Balance & Diversity Metrics

- **Total Recognized Classes:** `31`
- **Normalized Shannon Entropy:** `1.0` (1.0 = ideal balance)
- **Imbalance Ratio (Max/Min):** `1.0x`

## 6. Top Classes Breakdown

| Canonical Label | Samples | Type |
| :--- | :--- | :--- |
| `arecanut___healthy` | `4` | Validated |
| `arecanut___koleroga` | `4` | Validated |
| `arecanut___yellow_leaf` | `4` | Validated |
| `coconut___bud_rot` | `4` | Validated |
| `coconut___healthy` | `4` | Validated |
| `coconut___stem_bleeding` | `4` | Validated |
| `cotton___bacterial_blight` | `4` | Validated |
| `cotton___healthy` | `4` | Validated |
| `cotton___pink_bollworm` | `4` | Validated |
| `maize___common_rust` | `4` | Validated |
| `maize___fall_armyworm` | `4` | Validated |
| `maize___healthy` | `4` | Validated |
| `maize___northern_leaf_blight` | `4` | Validated |
| `ragi___blast` | `4` | Validated |
| `ragi___brown_spot` | `4` | Validated |
| `ragi___healthy` | `4` | Validated |
| `rice___bacterial_blight` | `4` | Validated |
| `rice___blast` | `4` | Validated |
| `rice___brown_spot` | `4` | Validated |
| `rice___healthy` | `4` | Validated |

