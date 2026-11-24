"""
Cloud GPU Training Pipeline for Multi-Task Karnataka Agriculture Models.
Features:
- PyTorch Mixed Precision (AMP / torch.cuda.amp)
- Multi-Task Loss: Focal / Cross-Entropy for Crop & Disease
- Cosine Annealing Learning Rate Scheduler
- Checkpoint saving to backend/models_store/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("AgriTrain")


def train_model(
    manifest_csv: str = "data_pipeline/manifests/unified_dataset_manifest.csv",
    output_dir: str = "backend/models_store",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-4,
    device_str: str = "cuda",
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        import torchvision.transforms as T
    except ImportError:
        logger.error("PyTorch and torchvision are required for training.")
        return

    from training.dataset import AgricultureDataset
    from training.model import HierarchicalAgriNet

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info("Using compute device: %s (%s)", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Data Augmentations
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AgricultureDataset(manifest_csv, split="train", transform=train_transform)
    val_ds = AgricultureDataset(manifest_csv, split="val", transform=val_transform,
                                crop_to_idx=train_ds.crop_to_idx, label_to_idx=train_ds.label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = HierarchicalAgriNet(
        num_crops=len(train_ds.crop_to_idx),
        num_conditions=len(train_ds.label_to_idx),
        pretrained=True,
    ).to(device)

    criterion_crop = nn.CrossEntropyLoss()
    criterion_cond = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_weights_path = out_path / "crop_disease_vision.pt"

    logger.info("Starting training across %d epochs ...", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_crop_correct, train_cond_correct, total_samples = 0.0, 0, 0, 0

        for images, crops, labels, _ in train_loader:
            images, crops, labels = images.to(device), crops.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                crop_logits, cond_logits = model(images)
                loss_crop = criterion_crop(crop_logits, crops)
                loss_cond = criterion_cond(cond_logits, labels)
                total_loss = loss_crop + 1.5 * loss_cond

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item() * len(images)
            train_crop_correct += (crop_logits.argmax(dim=-1) == crops).sum().item()
            train_cond_correct += (cond_logits.argmax(dim=-1) == labels).sum().item()
            total_samples += len(images)

        scheduler.step()

        # Validation loop
        model.eval()
        val_loss, val_crop_correct, val_cond_correct, val_samples = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, crops, labels, _ in val_loader:
                images, crops, labels = images.to(device), crops.to(device), labels.to(device)
                crop_logits, cond_logits = model(images)
                loss_crop = criterion_crop(crop_logits, crops)
                loss_cond = criterion_cond(cond_logits, labels)
                total_loss = loss_crop + 1.5 * loss_cond

                val_loss += total_loss.item() * len(images)
                val_crop_correct += (crop_logits.argmax(dim=-1) == crops).sum().item()
                val_cond_correct += (cond_logits.argmax(dim=-1) == labels).sum().item()
                val_samples += len(images)

        avg_train_loss = train_loss / max(total_samples, 1)
        avg_val_loss = val_loss / max(val_samples, 1)
        val_crop_acc = val_crop_correct / max(val_samples, 1)
        val_cond_acc = val_cond_correct / max(val_samples, 1)

        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f | Val Loss: %.4f | Val Crop Acc: %.1f%% | Val Disease Acc: %.1f%%",
            epoch, epochs, avg_train_loss, avg_val_loss, val_crop_acc * 100, val_cond_acc * 100
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "crop_to_idx": train_ds.crop_to_idx,
                "label_to_idx": train_ds.label_to_idx,
                "val_crop_acc": val_crop_acc,
                "val_cond_acc": val_cond_acc,
            }, best_weights_path)
            logger.info("  ✓ Best model saved to %s", best_weights_path)

    # Save class mappings JSON
    with open(out_path / "vision_class_indices.json", "w", encoding="utf-8") as f:
        json.dump({
            "crop_to_idx": train_ds.crop_to_idx,
            "label_to_idx": train_ds.label_to_idx,
            "idx_to_crop": train_ds.idx_to_crop,
            "idx_to_label": train_ds.idx_to_label,
        }, f, indent=2)

    logger.info("Training complete. Artifacts written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, device_str=args.device)
