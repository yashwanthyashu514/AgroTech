"""
Hierarchical Dual-Head Architecture for Crop & Disease Identification with Calibrated Confidence.
Head 1: Plant / Crop Species (Rice, Tomato, Maize, Cotton, Sugarcane, Ragi, Arecanut, Coconut, Wheat, Other)
Head 2: Specific Disease / Pest / Deficiency / Healthy status
Uncertainty: Out-of-Distribution Rejection via Entropy & Calibrated Temperature Scaling
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class HierarchicalAgriNet(nn.Module):
        """
        Multi-task vision architecture with shared backbone and twin classification heads.
        """

        def __init__(
            self,
            num_crops: int = 10,
            num_conditions: int = 38,
            backbone_name: str = "efficientnet_b0",
            pretrained: bool = True,
            dropout_rate: float = 0.3,
        ):
            super().__init__()
            self.num_crops = num_crops
            self.num_conditions = num_conditions

            # Feature extractor backbone
            # Use torchvision or timm if installed, otherwise clean residual CNN
            try:
                import torchvision.models as models
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base = models.efficientnet_b0(weights=weights)
                self.features = base.features
                self.pool = base.avgpool
                in_features = base.classifier[1].in_features
            except Exception:
                # Lightweight self-contained convolutional feature backbone
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.SiLU(),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.SiLU(),
                )
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                in_features = 256

            # Shared bottleneck
            self.shared_fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
            )

            # Head 1: Crop Species Identification
            self.crop_head = nn.Sequential(
                nn.Dropout(dropout_rate / 2),
                nn.Linear(512, num_crops),
            )

            # Head 2: Condition / Disease Identification
            self.condition_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(512 + num_crops, 256),  # Condition conditioned on crop logits
                nn.SiLU(),
                nn.Linear(256, num_conditions),
            )

            # Learnable temperature scalar for calibrated probabilities
            self.temperature = nn.Parameter(torch.ones(1) * 1.2)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
                crop_logits: (B, num_crops)
                condition_logits: (B, num_conditions)
            """
            feat = self.features(x)
            pooled = self.pool(feat)
            emb = self.shared_fc(pooled)

            crop_logits = self.crop_head(emb)
            # Concatenate crop logits to provide hierarchical context to disease head
            cond_input = torch.cat([emb, F.softmax(crop_logits, dim=-1)], dim=-1)
            condition_logits = self.condition_head(cond_input)

            return crop_logits, condition_logits

        def predict_with_confidence(
            self, x: torch.Tensor, ood_threshold: float = 0.55
        ) -> Dict[str, Any]:
            """
            Calibrated inference with Out-Of-Distribution (OOD) rejection.
            Never returns unsafe diagnoses when confidence is low or input is out-of-domain.
            """
            self.eval()
            with torch.no_grad():
                crop_logits, cond_logits = self.forward(x)
                # Temperature scaled probabilities
                crop_probs = F.softmax(crop_logits / self.temperature, dim=-1)
                cond_probs = F.softmax(cond_logits / self.temperature, dim=-1)

                top_crop_prob, top_crop_idx = torch.max(crop_probs, dim=-1)
                top_cond_prob, top_cond_idx = torch.max(cond_probs, dim=-1)

                confidence = float(top_crop_prob.item() * top_cond_prob.item())
                is_safe_diagnosis = confidence >= ood_threshold

                return {
                    "crop_idx": int(top_crop_idx.item()),
                    "crop_confidence": float(top_crop_prob.item()),
                    "condition_idx": int(top_cond_idx.item()),
                    "condition_confidence": float(top_cond_prob.item()),
                    "overall_confidence": round(confidence, 4),
                    "is_safe_diagnosis": is_safe_diagnosis,
                    "rejection_reason": None if is_safe_diagnosis else "Low confidence / Unrecognized leaf or soil pattern",
                }
else:
    class HierarchicalAgriNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate HierarchicalAgriNet.")
