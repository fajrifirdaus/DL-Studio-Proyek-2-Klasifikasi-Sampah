"""KANDIDAT: ensemble ConvNeXt V2-Tiny @384 (timm) + EfficientNet V2-S @384 (torchvision).

APA INI?
--------
Satu nn.Module berisi DUA model terlatih. Prediksi = rata-rata softmax keduanya
(bobot sama). Tidak ada pelatihan baru: bobot diambil dari dua run yang sudah ada,
    runs/convnextv2_tiny_384_split_seed42/   (macro-F1 my_val 0,9725)
    runs/efficientnet_v2_s_split_seed42/     (macro-F1 my_val 0,9618)
dan digabung jadi satu state_dict berawalan "cnx." dan "v2s." (lihat
experiments/buat_ensemble.py).

KENAPA DICOBA?
--------------
Pada my_val kedua model salah di gambar yang sebagian BERBEDA (McNemar: 11 vs 7
gambar), jadi rata-rata keduanya berpeluang menutup sebagian kesalahan.

KONTRAK SUBMISSION
------------------
  - get_model() hanya membangun arsitektur (tanpa unduh bobot).
  - get_transform() menghasilkan SATU tensor [3, 384, 384] untuk kedua model:
    keduanya memakai resolusi 384 dan normalisasi ImageNet yang sama.
  - Logits keluaran = log(rata-rata softmax) -> argmax sama dengan rata-rata
    probabilitas, bentuk [batch, 5].
  - Butuh timm (untuk ConvNeXt V2) + torchvision.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

NUM_CLASSES = 5
IMG_SIZE = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# Interpolasi resize: "bilinear" (cara V2-S dilatih) atau "bicubic" (cara ConvNeXt
# V2 dilatih). Dipilih berdasarkan uji di my_val (lihat experiments/README.md).
INTERPOLASI = "bicubic"   # bicubic 0,9672 vs bilinear 0,9650 pada my_val


class Ensemble(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnx = timm.create_model("convnextv2_tiny", pretrained=False,
                                     num_classes=NUM_CLASSES, drop_path_rate=0.1)
        self.v2s = efficientnet_v2_s(weights=None)
        self.v2s.classifier[1] = nn.Linear(self.v2s.classifier[1].in_features, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = (self.cnx(x).softmax(1) + self.v2s(x).softmax(1)) / 2
        return torch.log(p.clamp_min(1e-8))


def get_model() -> nn.Module:
    return Ensemble()


def get_transform():
    mode = (transforms.InterpolationMode.BICUBIC if INTERPOLASI == "bicubic"
            else transforms.InterpolationMode.BILINEAR)
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=mode, antialias=True),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
