"""CONTOH kandidat: ResNet-34 (bukan ResNet-18).

CARA PAKAI (di notebook, sel KONFIGURASI):
    CFG["arsitektur"] = "experiments/resnet34_model.py"
    CFG["backbone"]   = "resnet34"        # untuk unduh bobot ImageNet
    CFG["tag"]        = "coba_resnet34"   # supaya runs/ tidak saling menimpa

Lalu jalankan ulang sel 13 & 14 (data loader + LATIH).

KENAPA HARUS DUA BARIS (arsitektur + backbone)?
    - `arsitektur` menentukan BENTUK model (dipakai untuk melatih)
    - `backbone`   menentukan BOBOT ImageNet mana yang disalin
    Kalau `backbone` salah, bobotnya tidak akan cocok dan hampir tidak ada yang tersalin.

KALAU MENANG, SALIN KE submission/model.py DENGAN TANGAN:
    1. buka submission/model.py
    2. ganti isi get_model() dengan isi get_model() di sini
    3. PASTIKAN get_transform() di submission/model.py SAMA dengan di sini
       (resolusi 224 = Resize(256) + CenterCrop(224))
    4. latih ulang dengan CFG["tulis_submission"] = True supaya
       submission/model.pth cocok dengan model.py yang baru
       (model.py dan model.pth WAJIB berpasangan)

CATATAN JUJUR
    Ganti backbone BUKAN jaminan skor naik. Baseline resmi sudah 0,92 dan
    ruang perbaikannya lebih besar di cara melatih. ResNet-34 juga lebih berat:
    di T4 perkiraan 1,5-2x waktu latih ResNet-18, dan model.pth tumbuh dari
    +-44,7 MB jadi +-85 MB (masih di bawah batas 100 MB GitHub).
"""
import torch.nn as nn
from torchvision import models, transforms

# Nama backbone torchvision -> dipakai train.py untuk mengunduh bobot ImageNet.
# TANPA baris ini, bobot ImageNet TIDAK disalin (latih dari nol).
BACKBONE = "resnet34"

NUM_CLASSES = 5


def get_model():
    """Bangun ResNet-34 dengan kepala 5 kelas.

    PENTING: weights=None -> JANGAN mengunduh bobot di sini. Yang menyalin bobot
    ImageNet adalah train.py (lewat salin_bobot_imagenet), dan yang memuat
    model.pth adalah evaluate.py. Fungsi ini hanya membangun bentuknya.
    """
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transform():
    """Transform validasi. WAJIB sama persis dengan yang di submission/model.py
    supaya angka latih dan angka asisten bisa dibandingkan.

    Resolusi 224 = Resize(256) + CenterCrop(224) - sama seperti ResNet-18 kita.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
