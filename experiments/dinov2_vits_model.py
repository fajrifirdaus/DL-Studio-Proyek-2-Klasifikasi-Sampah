"""KANDIDAT: DINOv2 ViT-S/14 @224 (timm) - pretrained self-supervised LVD-142M.

    python train.py --data ./train --arsitektur experiments/dinov2_vits_model.py \
        --lr 2e-5 --weight-decay 0.05 --warmup 1 --clip-grad 1.0 --batch-size 16 \
        --tag dinov2_vits_224_lr2e-5_seed42

KENAPA DICOBA?
--------------
  - Fitur self-supervised dari 142 juta gambar terkurasi; dikenal kuat untuk
    dataset kecil (+-2.750 gambar).
  - 22,1 jt parameter -> bobot fp16 +-44 MB.

CATATAN
-------
1. Bawaan DINOv2 518x518 terlalu berat untuk GPU 6 GB dan waktu evaluasi;
   dipakai 224 (16x16 patch). Posisi embedding diinterpolasi dari 37x37 ke 16x16
   saat memuat bobot pretrained (resample_abs_pos_embed timm).
2. Model ini hanya dipretrain (belum di-fine-tune ImageNet) -> kepala 5 kelas
   dilatih dari nol bersama backbone.
3. Butuh timm. get_model() tidak mengunduh bobot.
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torchvision import transforms

BACKBONE = "dinov2_vits"
ARCH = "vit_small_patch14_dinov2"
TIMM_NAME = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE = 224
NUM_CLASSES = 5
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model() -> nn.Module:
    return timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES,
                             img_size=IMG_SIZE, drop_path_rate=0.1)


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def muat_bobot_pretrained(model: nn.Module) -> dict:
    # img_size=224 saat membuat model sumber -> timm otomatis menginterpolasi pos_embed
    src = timm.create_model(TIMM_NAME, pretrained=True, img_size=IMG_SIZE).state_dict()
    own = model.state_dict()
    transfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    dilewati = [k for k in src if k not in transfer]
    model.load_state_dict(transfer, strict=False)
    return {"backbone": TIMM_NAME, "disalin": len(transfer),
            "dilewati": dilewati, "total_kunci_model_kita": len(own)}
