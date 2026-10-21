"""KANDIDAT: CAFormer-S18 @384 (timm) - MetaFormer conv+attention, ImageNet-22k -> 1k.

    python train.py --data ./train --arsitektur experiments/caformer_s18_model.py \
        --epochs 15 --lr 5e-5 --weight-decay 0.05 --warmup 1 --clip-grad 1.0 \
        --start-epoch 6 --patience 5 --batch-size 8 --tag caformer_s18_384_lr5e-5_seed42

KENAPA DICOBA?
--------------
  - Pelajaran Preview 2: pretrain ImageNet-22k lebih menentukan daripada ukuran.
  - CAFormer = tahap awal konvolusi (seperti ConvNeXt) + tahap akhir self-attention
    (seperti ViT): menggabungkan dua keluarga yang sama-sama terbaik di percobaan kita
    (ConvNeXt V2-Tiny 0,9725 dan EVA-02 Small 0,9781).
  - 24,3 jt parameter -> bobot fp16 +-49 MB (< 100 MB, tanpa Release/LFS).

CATATAN: normalisasi ImageNet, bicubic 384. Butuh timm; get_model() tidak mengunduh bobot.
"""
from __future__ import annotations

import timm
import torch.nn as nn
from torchvision import transforms

BACKBONE = "caformer_s18"
ARCH = "caformer_s18"
TIMM_NAME = "caformer_s18.sail_in22k_ft_in1k_384"
IMG_SIZE = 384
NUM_CLASSES = 5
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_model() -> nn.Module:
    return timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES,
                             drop_path_rate=0.15)


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
