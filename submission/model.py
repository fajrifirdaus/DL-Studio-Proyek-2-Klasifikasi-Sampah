"""Contoh submission: ResNet-18 dengan head 5 kelas.

Ganti isi file ini dengan arsitektur kelompok Anda. Aturan:
  - get_model() hanya MEMBANGUN arsitektur (jangan mengunduh bobot pretrained di sini,
    jangan memuat model.pth di sini — evaluate.py yang memuatnya).
  - get_transform() harus SAMA dengan transform validasi saat Anda melatih model.
  - Output model: logits [batch, 5], urutan kelas:
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
    (urutan alfabetis — sama dengan torchvision.datasets.ImageFolder).
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ============================================================
# CATATAN UNTUK KELOMPOK (bukan bagian yang dinilai asisten)
# ============================================================
# Berkas ini adalah SALINAN dari contoh resmi. Kalau nanti kalian menemukan
# arsitektur yang lebih baik lewat experiments/, SALIN hasilnya ke sini:
#
#   1. ganti isi get_model() dengan versi kandidat
#   2. PASTIKAN get_transform() di sini SAMA dengan transform kandidat
#      (kalau resolusi latihnya 224, tetap Resize(256) + CenterCrop(224))
#   3. latih ulang supaya model.pth cocok:
#        di notebook : CFG["tulis_submission"] = True
#        di laptop   : python train.py --data "<...>/train"
#   4. uji: python evaluate.py --submission ./submission --data ./my_val
#
# PENTING: model.py dan model.pth WAJIB berpasangan. Kalau model.py jadi
# ResNet-34 tapi model.pth masih bobot ResNet-18, evaluasi asisten GAGAL
# (load_state_dict strict=True) dan skor putaran itu hilang.
# ============================================================
