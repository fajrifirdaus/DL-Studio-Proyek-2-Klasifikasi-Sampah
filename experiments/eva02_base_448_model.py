"""KANDIDAT: EVA-02 Base @448 (timm) - pretrained MIM ImageNet-22k -> fine-tune ImageNet-1k.

    python train.py --data ./train --arsitektur experiments/eva02_base_448_model.py \
        --lr 2e-5 --weight-decay 0.05 --warmup 1 --clip-grad 1.0 --batch-size 6 \
        --epochs 15 --patience 5 --start-epoch 6 --tag eva02_base_448_lr2e-5_seed42

KENAPA DICOBA?
--------------
  - Naik skala dari model yang sekarang terpasang (EVA-02 Small @336): 22,13 jt -> 87,12 jt
    parameter, 336 -> 448 px, dengan CARA PRETRAIN YANG SAMA (MIM ImageNet-22k, guru EVA-CLIP,
    lalu fine-tune penulis paper di ImageNet-1k). Jadi variabel yang berubah hanya
    ukuran/resolusi, bukan resep data pretrain.
  - ImageNet-1k top-1: Small 85,74 -> Base 88,26 (+2,5 poin). Di repo ini, backbone lain yang
    cuma +1,6-2 poin di atas Small (CAFormer-S18, TinyViT-21M, ConvNeXtV2-Base) hanya menambah
    0,0014-0,006 macro-F1 pada split kita, jadi kenaikannya TIDAK dijamin -- wajib diukur dulu
    pada split yang sama (2.340 latih / 413 val, seed 42) sebelum dipromosikan.

CATATAN
-------
1. Normalisasi CLIP (bukan ImageNet). MEAN/STD di bawah dibaca train.py juga untuk augmentasi
   latih, jadi latih dan evaluasi memakai normalisasi yang sama. Keluarga EVA-02 di timm
   memakai OPENAI_CLIP_MEAN/STD.
2. UKURAN BOBOT: fp32 = 348,5 MB, fp16 = +-174,2 MB.
     - batas submission : 500 MB  -> fp32 maupun fp16 LOLOS.
     - batas GitHub     : 100 MB  -> LEWAT, jadi saat dipromosikan WAJIB lampirkan sebagai
       aset GitHub Release atau Git LFS + tulis tautannya di README.
3. RESOLUSI 448 px: train.py otomatis menurunkan batch ke 6 (tabel _batas_batch untuk T4 16 GB).
   Jangan dipaksa lebih besar. Kebutuhan memori +-1,8x dari 336 px.
4. RESOLUSI 448 px bersifat MENGIKAT: patch_embed EVA-02 (base/small) TIDAK memakai
   dynamic_img_size, jadi timm menolak input selain 448 dengan pesan
   "Input height (X) doesn't match model (448)". Latih dan evaluasi WAJIB sama-sama 448.
   (Kalau ingin resolusi lain: create_model(..., dynamic_img_size=True) lalu LATIH ULANG
   di resolusi itu -- jangan cuma mengubah get_transform().)
5. Butuh timm. get_model() TIDAK mengunduh bobot; bobot ImageNet hanya diunduh saat latih
   lewat muat_bobot_pretrained() (+-348 MB, sekali, perlu internet).
"""
from __future__ import annotations

# ============================================================
# KONFIGURASI
# ============================================================
BACKBONE = "eva02_base"                                     # label di config.json/metrics.json
ARCH = "eva02_base_patch14_448"                             # nama arsitektur timm
TIMM_NAME = "eva02_base_patch14_448.mim_in22k_ft_in1k"      # bobot: MIM in22k -> ft ImageNet-1k
TIMM_NAME_22K = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"  # alternatif (lihat catatan 6)
IMG_SIZE = 448                                              # resolusi latih = resolusi transform
NUM_CLASSES = 5                                             # botol/gelas/kertas/saset/styrofoam

# Normalisasi CLIP - SAMA dengan EVA-02 Small supaya perbandingan apple-to-apple.
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

# Catatan 6: TIMM_NAME_22K punya top-1 88,69 tetapi diukur pada 21.841 kelas ImageNet-22k, bukan
# 1.000 kelas; angka itu BUKAN berarti "lebih pintar" dari TIMM_NAME (88,26 pada 1.000 kelas).
# Kalau tetap ingin mencoba yang 22k, cukup ganti satu baris di muat_bobot_pretrained().
# ============================================================

import timm
import torch.nn as nn
from torchvision import transforms


def get_model() -> nn.Module:
    """Bangun arsitektur saja (pretrained=False -> tanpa unduh bobot).

    drop_path_rate 0,1 disamakan dengan EVA-02 Small. Kalau nanti terbukti overfit
    (train acc jauh di atas val), boleh dinaikkan ke 0,2 dan ulangi run.
    """
    return timm.create_model(ARCH, pretrained=False, num_classes=NUM_CLASSES,
                             drop_path_rate=0.1)


def get_transform():
    """Transform validasi: Resize 448 bicubic -> CenterCrop 448 -> ToTensor -> norm CLIP.

    Resolusi ini WAJIB sama dengan resolusi latih (train.py memeriksanya lewat
    get_transform() dan menyesuaikan augmentasi latih otomatis).
    """
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def muat_bobot_pretrained(model: nn.Module) -> dict:
    """Salin bobot pretrained hanya untuk tensor yang bentuknya cocok.

    Kepala klasifikasi (1.000 kelas ImageNet) tidak ikut karena jumlah kelas kita 5:
    tensor itu otomatis masuk daftar "dilewati" lewat perbandingan bentuk.
    """
    src = timm.create_model(TIMM_NAME, pretrained=True).state_dict()
    own = model.state_dict()
    transfer = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    dilewati = [k for k in src if k not in transfer]
    model.load_state_dict(transfer, strict=False)
    return {"backbone": TIMM_NAME, "disalin": len(transfer),
            "dilewati": dilewati, "total_kunci_model_kita": len(own)}
