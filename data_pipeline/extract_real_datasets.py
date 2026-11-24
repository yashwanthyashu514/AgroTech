"""
Extracts real images from downloaded agricultural datasets (PlantDoc, Rice Leaf Disease, etc.)
and organizes them for training and inference verification.
"""
import io
import pathlib
import pandas as pd
from PIL import Image

out_base = pathlib.Path("data_pipeline/real_datasets")
out_base.mkdir(parents=True, exist_ok=True)

# 1. Extract PlantDoc samples
plantdoc_p = pathlib.Path("scratch/plantdoc_test/data/test-00000-of-00001.parquet")
if plantdoc_p.exists():
    df = pd.read_parquet(plantdoc_p)
    count = 0
    for idx, row in df.iterrows():
        img_bytes = row["image"]["bytes"]
        objs = row["objects"]
        if len(objs["category"]) > 0:
            cat = str(objs["category"][0]).replace(" ", "_").replace("/", "_")
            cat_dir = out_base / f"PlantDoc_{cat}"
            cat_dir.mkdir(parents=True, exist_ok=True)
            img_path = cat_dir / f"img_{idx:04d}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            count += 1
    num_cats = len(list(out_base.glob("PlantDoc_*")))
    print(f"Extracted {count} real PlantDoc field images across {num_cats} categories")

# 2. Extract Rice disease samples
rice_p = pathlib.Path("scratch/rice_test/data/diseases-00000-of-00002.parquet")
if rice_p.exists():
    df_rice = pd.read_parquet(rice_p)
    rice_count = 0
    for lbl in ["Bacterial Leaf Blight", "Narrow Brown Spot", "Leaf Scald"]:
        subset = df_rice[df_rice["label"] == lbl].head(40)
        c_dir = out_base / f"Rice_{lbl.replace(' ', '_')}"
        c_dir.mkdir(parents=True, exist_ok=True)
        for i, (_, row) in enumerate(subset.iterrows()):
            img_bytes = row["image"]["bytes"]
            with open(c_dir / f"rice_{i:03d}.jpg", "wb") as f:
                f.write(img_bytes)
            rice_count += 1
    print(f"Extracted {rice_count} real Rice disease field images")

# 3. Add user test images
user_chilli_src = pathlib.Path("data_pipeline/vision_training_data/Chilli___healthy_mature/user_chilli_real.jpg")
if user_chilli_src.exists():
    chilli_out = out_base / "User_Chilli_Real"
    chilli_out.mkdir(parents=True, exist_ok=True)
    with open(user_chilli_src, "rb") as f_in, open(chilli_out / "chilli_001.jpg", "wb") as f_out:
        f_out.write(f_in.read())

user_tomato_src = pathlib.Path("scratch/user_tomato_crop.jpg")
if user_tomato_src.exists():
    tomato_out = out_base / "User_Tomato_Real"
    tomato_out.mkdir(parents=True, exist_ok=True)
    with open(user_tomato_src, "rb") as f_in, open(tomato_out / "tomato_001.jpg", "wb") as f_out:
        f_out.write(f_in.read())

user_leaf_src = pathlib.Path("scratch/user_leaf_crop.jpg")
if user_leaf_src.exists():
    leaf_out = out_base / "User_Diseased_Leaf_Real"
    leaf_out.mkdir(parents=True, exist_ok=True)
    with open(user_leaf_src, "rb") as f_in, open(leaf_out / "diseased_leaf_001.jpg", "wb") as f_out:
        f_out.write(f_in.read())

print("✅ Real datasets setup and extracted successfully!")
