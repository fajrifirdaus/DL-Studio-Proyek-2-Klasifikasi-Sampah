"""KANDIDAT: EfficientNet V2-S (efficientnet_v2_s) - tanpa timm.

APA INI?
--------
Cetakan (blueprint) model, BUKAN model terlatih. Dipanggil oleh train.py atau
notebook lewat CFG["arsitektur"], lalu dilatih. Hasil latihannya masuk
runs/<tag>.pth. Kalau menang, baru dipindah ke submission/.

KENAPA V2-S MENARIK?
--------------------
  - Ukuran bobot ImageNet 82,7 MB -> AMAN di bawah batas repo 100 MB
    (lebih kecil dari ResNet-34 yang 83,3 MB, dan ResNet-50 yang 97,8 MB)
  - EfficientNet V2 dilatih dengan cara yang lebih modern daripada V1,
    jadi biasanya lebih cepat konvergen dan lebih baik pada resolusi besar
  - Tersedia di torchvision -> SYARAT "tanpa timm" tetap terpenuhi

HARAP DIPERHATIKAN (JANGAN DILEWATKAN)
--------------------------------------
1. RESOLUSI 384, BUKAN 224.
   Semua EfficientNet V2 di torchvision disarankan crop_size=384. Kalau dipaksa
   224, akurasinya bisa justru LEBIH BURUK daripada ResNet-18. Berkas ini sudah
   menyetel get_transform() ke 384, dan train.py/notebook akan otomatis
   menyesuaikan augmentasi latihnya supaya cocok.

2. LATIH JADI LEBIH LAMA & LEBIH BERAT.
   Gambar 384x384 butuh sekitar 3x perhitungan dibanding 224x224. Perkiraan:
       ResNet-18 @224 : ~15 menit (T4)
       V2-S     @384 : ~50-60 menit (T4)   <-- naik ~3,5x
   Karena itu setel CFG["batch_size"] = 8 atau 16 (bukan 32) supaya T4 tidak
   kehabisan memori (OOM). Kalau OOM, turunkan lagi ke 8.

3. INI BELUM TENTU MENANG.
   Validasi kita hanya 413 gambar, jadi ambang beda nyata (perkiraan kasar
   1,96 x akar(0,25/n)) sekitar +-0,048 (4,8%). Selisih di bawah itu tidak bisa
   dibedakan dari keberuntungan. Karena itu latih dan uji beberapa seed/backbone,
   jangan menyimpulkan dari satu angka.
   Di repo referensi, EfficientNet V1 malah KALAH 6,7 poin dari ResNet-18.
   Jadi perlakukan ini sebagai percobaan, bukan peningkatan yang pasti.

CARA PAKAI (di notebook, sel KONFIGURASI)
-----------------------------------------
    CFG["arsitektur"] = "experiments/efficientnet_v2s_model.py"
    CFG["backbone"]   = "efficientnet_v2_s"   # untuk menyalin bobot ImageNet
    CFG["batch_size"] = 16                    # 384x384 lebih berat
    CFG["tag"]        = "coba_v2s"

CARA PAKAI (di laptop, kalau ada GPU)
-------------------------------------
    python train.py --data "<...>/train" \\
        --arsitektur experiments/efficientnet_v2s_model.py \\
        --backbone efficientnet_v2_s --batch-size 16 --tag coba_v2s

CATATAN: nama backbone di torchvision adalah "efficientnet_v2_s" (huruf kecil,
pakai garis bawah). Ini BEDA dari EfficientNet V1 yang bernama "efficientnet_b0".
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

# ============================================================
# KONFIGURASI
# ============================================================
# Nama backbone di torchvision - dipakai train.py untuk menyalin bobot ImageNet.
BACKBONE = "efficientnet_v2_s"

NUM_CLASSES = 5          # botol, gelas, kertas, saset, styrofoam (urut alfabetis)

# Resolusi yang disarankan V2-S. WAJIB sama antara get_model() dan get_transform().
IMG_SIZE = 384

# Normalisasi ImageNet (sama untuk semua model torchvision).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# ARSITEKTUR
# ============================================================
def get_model() -> nn.Module:
    """Kembalikan EfficientNet V2-S dengan kepala 5 kelas.

    weights=None WAJIB: kontrak submission/dan evaluation melarang mengunduh bobot
    di dalam get_model(). Bobot ImageNet disalin terpisah oleh train.py lewat
    salin_bobot_imagenet(), memakai nama BACKBONE di atas.
    """
    model = efficientnet_v2_s(weights=None)

    # Kepala EfficientNet ada di model.classifier (bukan model.fc seperti ResNet).
    # classifier = Sequential(Dropout, Linear(1280 -> 1000)).
    # Kita ganti Linear-nya supaya keluar 5 kelas. Bentuk parameter barunya
    # (5, 1280) berbeda dari ImageNet (1000, 1280), jadi salin bobot otomatis
    # MELEWATI kepala ini - dan itu memang yang diinginkan.
    masuk = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(masuk, NUM_CLASSES)

    return model


# ============================================================
# TRANSFORM (dipakai evaluation, jadi WAJIB bebas acak)
# ============================================================
def get_transform():
    """Transform untuk validasi/evaluasi - TANPA keacakan.

    Resize(384) TANPA CenterCrop tambahan, karena crop_size V2-S di torchvision
    juga 384 (resize dan crop berukuran sama).
    """
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, antialias=True),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
