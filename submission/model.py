"""KANDIDAT: EVA-02 Small @336 (timm) - pretrained MIM ImageNet-22k -> fine-tune ImageNet-1k.

    python train.py --data ./train --arsitektur experiments/eva02_small_model.py \
        --lr 2e-5 --weight-decay 0.05 --warmup 1 --clip-grad 1.0 --batch-size 8 \
        --tag eva02_small_336_lr2e-5_seed42

KENAPA DICOBA?
--------------
  - Pelajaran dari ConvNeXt V2: data/cara pretrain lebih menentukan daripada ukuran
    model (Base 87,7 jt hanya +0,0014 atas Tiny 27,9 jt).
  - EVA-02 dipretrain Masked Image Modeling dengan target fitur CLIP di
    ImageNet-22k -> representasi kuat, tetapi hanya 21,7 jt parameter.
  - Bobot fp16 +-43 MB -> bisa di-commit langsung (< 100 MB).

CATATAN
-------
1. Normalisasi CLIP (bukan ImageNet). MEAN/STD di bawah dibaca train.py juga untuk
   augmentasi latih, jadi latih dan evaluasi memakai normalisasi yang sama.
2. ViT -> lr lebih kecil dari ConvNeXt (2e-5).
3. Butuh timm. get_model() tidak mengunduh bobot.
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torchvision import transforms

BACKBONE = "eva02_small"
ARCH = "eva02_small_patch14_336"
TIMM_NAME = "eva02_small_patch14_336.mim_in22k_ft_in1k"
IMG_SIZE = 336
NUM_CLASSES = 5
MEAN = (0.48145466, 0.4578275, 0.40821073)     # normalisasi CLIP
STD = (0.26862954, 0.26130258, 0.27577711)


def get_model() -> nn.Module:
    return timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES,
                             drop_path_rate=0.1)


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
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
