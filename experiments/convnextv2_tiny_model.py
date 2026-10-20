"""KANDIDAT: ConvNeXt V2-Tiny @384 (timm) - pretrained FCMAE + ImageNet-22k -> 1k.

APA INI?
--------
Cetakan (blueprint) model untuk dicoba lewat train.py:
    python train.py --data ./train --arsitektur experiments/convnextv2_tiny_model.py \
        --tag convnextv2_tiny_384_split_seed42
Bobot hasil latih masuk runs/<tag>/<tag>.pth. Kalau menang, dipindah ke
submission/ lewat experiments/promote.py.

KENAPA DICOBA?
--------------
  - Dipretrain FCMAE lalu di-fine-tune ImageNet-22k -> 1k: representasi lebih
    kaya untuk dataset kecil (+-2.750 gambar) dibanding bobot ImageNet-1k biasa.
  - 27,9 juta parameter (V2-S 20,2 juta) -> bobot fp32 +-112 MB.
  - Memakai LayerNorm (bukan BatchNorm) -> stabil dengan batch kecil (8 @384).

CATATAN PENTING
---------------
1. BUTUH timm. Diizinkan aturan studio ("backbone pretrained torchvision / timm")
   dan timm termasuk paket dasar asisten. Bila dipasang ke submission/,
   tambahkan `timm` ke submission/requirements.txt.
2. get_model() TIDAK mengunduh bobot (pretrained=False) sesuai kontrak.
   Bobot ImageNet hanya diambil saat LATIH, lewat muat_bobot_pretrained()
   yang dipanggil train.py.
3. Bobot > 100 MB -> bila dipasang, model.pth wajib lewat GitHub Release / LFS
   atau disimpan fp16 (lihat experiments/README.md).
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torchvision import transforms

BACKBONE = "convnextv2_tiny"            # label untuk nama run/tag
TIMM_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
IMG_SIZE = 384
NUM_CLASSES = 5
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model() -> nn.Module:
    """Arsitektur saja (tanpa unduh bobot). Kepala 5 kelas."""
    return timm.create_model(TIMM_NAME, pretrained=False,
                             num_classes=NUM_CLASSES, drop_path_rate=0.1)


def get_transform():
    """PIL RGB -> Tensor [3, 384, 384]; sama dengan konfigurasi data timm (crop_pct 1.0)."""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def muat_bobot_pretrained(model: nn.Module) -> dict:
    """Dipanggil train.py saat latih: salin bobot timm yang bentuknya cocok (kepala dilewati)."""
    src = timm.create_model(TIMM_NAME, pretrained=True).state_dict()
    own = model.state_dict()
    transfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    dilewati = [k for k in src if k not in transfer]
    model.load_state_dict(transfer, strict=False)
    return {"backbone": TIMM_NAME, "disalin": len(transfer),
            "dilewati": dilewati, "total_kunci_model_kita": len(own)}
