"""KANDIDAT: ConvNeXt V2-Base @224 (timm) - pretrained FCMAE + ImageNet-22k -> 1k.

Saran rekan kelompok. Dicoba SETELAH ConvNeXt V2-Tiny sebagai pembanding ukuran.

    python train.py --data ./train --arsitektur experiments/convnextv2_base_model.py \
        --batch-size 16 --tag convnextv2_base_224_split_seed42

CATATAN PENTING
---------------
1. 87,7 juta parameter -> bobot fp32 +-350 MB (> 100 MB GitHub, < 500 MB aturan).
   Bila dipasang: lampirkan via GitHub Release / Git LFS + tautan di README.
2. Resolusi 224 (bukan 384): pada 384 biaya hitung ~3x dan memori GPU 6 GB tidak
   cukup; 224 juga menjaga waktu evaluate.py jauh di bawah 15 menit.
3. Butuh timm (diizinkan aturan studio). get_model() tidak mengunduh bobot.
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torchvision import transforms

BACKBONE = "convnextv2_base"
TIMM_NAME = "convnextv2_base.fcmae_ft_in22k_in1k"
IMG_SIZE = 224
RESIZE = 256                    # crop_pct 0.875 seperti konfigurasi timm
NUM_CLASSES = 5
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model() -> nn.Module:
    return timm.create_model(TIMM_NAME, pretrained=False,
                             num_classes=NUM_CLASSES, drop_path_rate=0.2)


def get_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def muat_bobot_pretrained(model: nn.Module) -> dict:
    src = timm.create_model(TIMM_NAME, pretrained=True).state_dict()
    own = model.state_dict()
    transfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    dilewati = [k for k in src if k not in transfer]
    model.load_state_dict(transfer, strict=False)
    return {"backbone": TIMM_NAME, "disalin": len(transfer),
            "dilewati": dilewati, "total_kunci_model_kita": len(own)}
