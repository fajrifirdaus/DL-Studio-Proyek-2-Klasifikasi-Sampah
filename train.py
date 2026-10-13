"""train.py - pelatihan ResNet-18 untuk Studio Proyek 2 (Klasifikasi Sampah).

Skrip ini MEMPERBAIKI celah pada `train_baseline.py` bawaan starter kit, tetapi
TETAP memakai ResNet-18 pretrained ImageNet (backbone tidak diganti).

TIGA PERBAIKAN YANG PALING BERPENGARUH (sisanya kecil):
    (a) seleksi model memakai MACRO-F1, bukan akurasi  <- yang dinilai asisten
    (b) pembobotan kelas, sehingga kelas minoritas (kertas_kardus) tidak kalah
    (c) 30 epoch (baseline hanya 8) + early stopping

APA SAJA CELAH YANG DITUTUP?
  1. Pembulatan split - baseline SEBENARNYA SUDAH berstrata per kelas (ambil
     int(n x 0.15) tiap kelas), jadi ini bukan celah besar. Bedanya hanya
     pembulatan: kita memakai max(1, round(n x 0.15)) sehingga tiap kelas dapat
     15% lebih tepat (413 vs 410 gambar validasi, selisih 3 gambar).
  2. Split kedaluwarsa - folder my_val/ dari latihan lama dipakai ulang tanpa
     dicek. Sekarang ada penanda .split.json; kalau seed/frac berubah, split
     otomatis dibangun ulang.
  3. Seleksi model    - baseline memilih model terbaik berdasarkan akurasi,
     padahal yang dinilai adalah MACRO-F1. Sekarang seleksi memakai macro-F1.
  4. Bobot kelas      - baseline tanpa pembobotan, sehingga kelas minoritas
     (kertas_kardus) kalah. Sekarang bobot kelas = N / (C * n_c).
  5. Early stopping   - baseline tidak punya. Sekarang ada patience 8 +
     start_epoch 10 (early stopping baru aktif setelah pemanasan). Model TERBAIK
     disimpan sejak epoch 1 tanpa ambang, jadi model bagus di awal tidak terbuang.
  6. Augmentasi       - resep latih disamakan dengan acuan yang terbukti
     (crop 0.60-1.00, flip, ColorJitter 0.3/0.3/0.3/0.05, RandomErasing 0.25).
     Rotasi dan crop sempit dari versi lama DIBUANG karena tidak terbukti
     menolong dan menambah waktu latih.
  7. Kontrak transform - resolusi get_transform() dicek otomatis terhadap
     resolusi latih, supaya tidak ada penurunan akurasi diam-diam.
  8. Salin bobot      - baseline meng-hardcode "fc." dari resnet18. Sekarang
     salinannya berbasis kecocokan bentuk tensor + dilaporkan.
  9. Verifikasi       - training mengevaluasi memakai get_model()/get_transform()
     dari submission/model.py, jadi mismatch kontrak ketahuan SEBELUM tenggat,
     bukan saat asisten menjalankan evaluate.py.
 10. Strategi RPS     - mendukung freezing / linear probing vs full fine-tuning,
    5-fold cross-validation, dan pelatihan ulang di 100% data.

CARA PAKAI (jalankan dari root repo, di mesin yang ada GPU):

    # 1) Latih standar (resnet18 pretrained, 30 epoch, seleksi macro-F1)
    python train.py --data "/path/ke/Dataset Sampah Studio 2/train"

    # 2) Uji sendiri dengan skrip resmi (WAJIB sebelum tenggat)
    python evaluate.py --submission ./submission --data ./my_val

    # 3) Uji cepat 1 menit dulu (memastikan skrip jalan sebelum latih penuh)
    python train.py --data ./train --limit 300 --epochs 2

    # 4) Eksperimen strategi fine-tuning (diminta RPS Studio 2)
    python train.py --data ./train --freeze-epochs 5     # linear probing -> full
    python train.py --data ./train --scratch             # CNN from scratch (P4)

    # 5) Mencoba backbone lain (kandidat dari experiments/)
    #    ResNet-34, resolusi 224, 83,3 MB:
    python train.py --data ./train --arsitektur experiments/resnet34_model.py \
                    --backbone resnet34 --tag coba_resnet34

    #    EfficientNet V2-S, resolusi 384, 82,7 MB (latih ~3,5x lebih lama):
    python train.py --data ./train --arsitektur experiments/efficientnet_v2s_model.py \
                    --backbone efficientnet_v2_s --batch-size 16 --tag coba_v2s

    # 6) 5-fold CV untuk memilih konfigurasi tanpa "mengintip" validasi
    python train.py --data ./train --kfolds 5 --fold 1

    # 7) Latih ulang di 100% data (langkah terakhir sebelum commit Final)
    python train.py --data ./train --full-data --epochs 18

KE MANA BOBOT DISIMPAN?
    Percobaan (bawaan) : runs/<tag>.pth
        -> tiap percobaan punya berkas SENDIRI, tidak saling menimpa, dan
           submission/ tetap aman.
    Putaran Final      : tambahkan --to-submission
        -> bobot masuk submission/model.pth (yang dinilai asisten).

Hasil setiap run disimpan di runs/<tag>/ (config, history, metrics) sebagai
bukti eksperimen untuk README.

PENTING: model.py dan model.pth WAJIB berpasangan. Kalau model.py jadi
ResNet-34 tetapi model.pth masih bobot ResNet-18, evaluasi asisten GAGAL
(load_state_dict strict=True).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from submission.model import get_model, get_transform

# ============================================================
# KONFIGURASI DASAR
# ============================================================
# Urutan kelas WAJIB sama dengan evaluate.py dan ImageFolder (alfabetis).
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus",
           "saset_kemasan", "styrofoam"]
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ============================================================
# UTILITAS UMUM
# ============================================================
def seed_everything(seed: int) -> None:
    """Tetapkan seed di semua sumber acak supaya hasil bisa diulang."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pilih perangkat: GPU (cuda) > Apple (mps) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmentasi untuk data latih pada resolusi `img_size`.

    RESEP INI SENGAJA DISAMAKAN dengan acuan yang sudah terbukti pada dataset
    yang sama (ResNet-18, 15 epoch -> macro-F1 0,9373). Tiga hal yang DIBUANG
    dari versi sebelumnya karena tidak terbukti menolong:

      - RandomResizedCrop: skala 0.60-1.00 (versi lama 0.70-1.00). Rentang lebih
        lebar -> model tidak sempat menghafal latar belakang, dan pada dataset
        kecil (2.753 gambar) itu menolong. Argumen `ratio` juga dibuang karena
        acuan memakai bawaan.
      - RandomRotation DIBUANG. Menambah waktu latih tanpa bukti menaikkan
        macro-F1; acuan yang tembus 0,9373 tidak memakainya.
      - ColorJitter dinaikkan ke 0.3/0.3/0.3/0.05 dan RandomErasing ke p=0.25
        (versi lama 0.3/0.3/0.2/0.02 dan p=0.20) supaya setara acuan.

    RandomHorizontalFlip DIPERTAHANKAN: kelima kelas (botol, gelas, kertas,
    saset, styrofoam) tidak punya ciri yang berubah makna bila dicerminkan.
    Ciri pembeda "berleher botol" vs "tanpa leher" tetap berlaku setelah
    dicerminkan. (Beda dengan dataset isyarat tangan, di mana flip merusak.)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.60, 1.00)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


# ============================================================
# MEMBACA DATASET DAN MEMBUAT SPLIT
# ============================================================
def list_dataset(data_dir: Path) -> list:
    """Baca folder <data_dir>/<nama_kelas>/*.jpg -> list berisi list (path, label).

    Urutan kelas mengikuti CLASSES (alfabetis), sama seperti ImageFolder.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"[GAGAL] folder data tidak ditemukan: {data_dir}")

    ada = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if set(ada) != set(CLASSES):
        raise SystemExit(
            "[GAGAL] nama folder kelas tidak sesuai.\n"
            f"  diharapkan : {CLASSES}\n"
            f"  ditemukan  : {ada}\n"
            "Pastikan --data menunjuk ke folder 'train/' dari dataset."
        )

    per_kelas = []
    for label, nama in enumerate(CLASSES):
        files = sorted(p for p in (data_dir / nama).iterdir()
                       if p.suffix.lower() in EXTS)
        if not files:
            raise SystemExit(f"[GAGAL] tidak ada gambar di kelas {nama}")
        per_kelas.append([(p, label) for p in files])
    return per_kelas


def limit_per_class(per_kelas: list, limit: int) -> list:
    """Ambil sebagian data saja (untuk uji cepat). limit = total gambar."""
    if not limit:
        return per_kelas
    per = max(1, limit // len(per_kelas))
    return [items[:per] for items in per_kelas]


def split_single(per_kelas: list, val_frac: float, seed: int) -> tuple:
    """Split tunggal BERSTRATA: tiap kelas diambil val_frac untuk validasi.

    Return: (train_items, val_items) - keduanya list (path, label).
    """
    rng = random.Random(seed)
    tr, va = [], []
    for items in per_kelas:
        items = list(items)
        rng.shuffle(items)
        n_va = max(1, int(round(len(items) * val_frac)))
        va += items[:n_va]
        tr += items[n_va:]
    return tr, va


def split_kfold(per_kelas: list, kfolds: int, fold: int, seed: int) -> tuple:
    """Split K-fold BERSTRATA. fold dimulai dari 1.

    Tiap kelas dibagi rata menjadi kfolds bagian; bagian ke-`fold` menjadi
    validasi, sisanya latih. Union semua fold validasi = seluruh dataset.
    """
    if not 1 <= fold <= kfolds:
        raise SystemExit(f"[GAGAL] --fold harus 1..{kfolds}, diberikan {fold}")
    rng = random.Random(seed)
    tr, va = [], []
    for items in per_kelas:
        items = list(items)
        rng.shuffle(items)
        for i, item in enumerate(items):
            (va if i % kfolds == fold - 1 else tr).append(item)
    return tr, va


def tulis_my_val(val_items: list, out_dir: Path, meta: dict) -> None:
    """Salin gambar validasi ke folder my_val/ supaya bisa diuji evaluate.py.

    Kalau penanda .split.json sudah ada dan isinya sama, penyalinan dilewati.
    Kalau berbeda (seed/frac berubah), folder dibangun ulang - ini yang
    menutup celah 'split kedaluwarsa' pada baseline.
    """
    out_dir = Path(out_dir)
    marker = out_dir / ".split.json"

    if marker.exists():
        try:
            lama = json.loads(marker.read_text())
        except Exception:
            lama = None
        if lama == meta:
            print(f"[split] my_val/ sudah sesuai (dilewati): {out_dir}")
            return
        print("[split] penanda my_val/ berbeda -> membangun ulang")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for path, label in val_items:
        d = out_dir / CLASSES[label]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, d / Path(path).name)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(meta, indent=2))
    print(f"[split] my_val/ dibuat: {out_dir} ({len(val_items)} gambar)")


# ============================================================
# DATASET DARI DAFTAR PATH
# ============================================================
class PathListDataset(Dataset):
    """Dataset sederhana dari daftar (path, label) + transform."""

    def __init__(self, items: list, tf):
        self.items = items
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


# ============================================================
# METRIK (implementasi harus SAMA dengan evaluate.py milik asisten)
# ============================================================
def macro_f1(y_true: list, y_pred: list, n_classes: int = NUM_CLASSES) -> tuple:
    """Hitung macro-F1, akurasi, confusion matrix, dan F1 per kelas.

    Rumusnya disamakan persis dengan evaluate.py bawaan (tanpa sklearn, karena
    paket dasar asisten hanya torch/torchvision/timm/pillow/numpy).
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1_list = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_list.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)

    acc = int(np.trace(cm)) / max(1, len(y_true))
    return float(np.mean(f1_list)), float(acc), cm, f1_list


def class_weight_tensor(labels: list, device) -> tuple:
    """Bobot kelas = N / (C * n_c). Dihitung HANYA dari data latih.

    Catatan penting (sering salah): yang sama dengan 1,0 adalah **rata-rata
    tertimbang frekuensi**, bukan rata-rata biasa:

        sum_c (n_c / N) * w_c = 1,0     <- SELALU benar
        mean_c (w_c)          = 1,0     <- HANYA benar kalau kelas seimbang

    Untuk dataset ini (rasio kelas 1,84x) rata-rata biasa = 1,051.
    Kalau rata-rata biasa dipakai sebagai pemeriksaan "harus 1,0", hasilnya
    akan terlihat seperti kesalahan padahal rumusnya sudah benar.
    """
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(np.float64)
    counts_aman = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (NUM_CLASSES * counts_aman)
    return torch.tensor(w, dtype=torch.float32, device=device), counts


# ============================================================
# MODEL: SALIN BOBOT IMAGENET + STRATEGI FREEZING
# ============================================================
def head_param_ids(model) -> set:
    """Kumpulkan id parameter kepala (fc / classifier) untuk keperluan freezing."""
    for nama in ("fc", "classifier"):
        head = getattr(model, nama, None)
        if head is not None:
            return {id(p) for p in head.parameters()}
    return set()


def set_backbone_trainable(model, trainable: bool) -> list:
    """Buka/tutup gradien backbone; kepala (head) selalu dilatih.

    PENTING - mengembalikan daftar parameter yang BARU dibuka. Parameter itu
    WAJIB dimasukkan ke optimizer lewat `opt.add_param_group(...)`. Tanpa itu,
    gradiennya dihitung tetapi dibuang, sehingga backbone seolah "dibuka" tapi
    tidak pernah dilatih. (Ini bug yang membuat akurasi mentok: optimizer
    terlanjur dibuat hanya dengan kepala.)
    """
    head_ids = head_param_ids(model)
    baru = []
    for p in model.parameters():
        mau = bool(trainable) or (id(p) in head_ids)
        if mau and not p.requires_grad:
            baru.append(p)
        p.requires_grad = mau
    return baru


def salin_bobot_imagenet(model, backbone: str) -> dict:
    """Salin bobot ImageNet hanya untuk tensor yang bentuknya cocok.

    Ini menggantikan cara baseline yang meng-hardcode awalan "fc." (khusus
    ResNet). Pendekatan berbasis bentuk tensor aman untuk backbone apa pun,
    dan melaporkan berapa tensor yang gagal disalin.
    """
    bobot = models.get_model_weights(backbone)["IMAGENET1K_V1"]
    try:
        src = models.get_model(backbone, weights=bobot, progress=True)
    except Exception as e:
        raise SystemExit(
            "[GAGAL] bobot ImageNet tidak bisa diunduh.\n"
            f"  sebab   : {type(e).__name__}: {e}\n"
            "  solusi  : (1) pastikan internet hidup, atau\n"
            "            (2) unduh bobot sekali di mesin lain lalu salin ke\n"
            "                ~/.cache/torch/hub/checkpoints/, atau\n"
            "            (3) jalankan dengan --scratch (tanpa bobot pretrained)."
        )
    pre, own = src.state_dict(), model.state_dict()

    transfer, dilewati = {}, []
    for k, v in pre.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            transfer[k] = v
        else:
            dilewati.append(k)

    model.load_state_dict(transfer, strict=False)
    return {"backbone": backbone, "disalin": len(transfer),
            "dilewati": dilewati, "total_kunci_model_kita": len(own)}


# ============================================================
# ARSITEKTUR PILIHAN SENDIRI (dari experiments/)
# ============================================================
def muat_arsitektur(jalur):
    """Impor get_model()/get_transform() dari sebuah BERKAS (bukan paket).

    Dipakai kalau kalian mau mencoba arsitektur/backbone lain tanpa mengubah
    submission/model.py. Berkasnya taruh di experiments/, contoh:

        python train.py --data "<...>/train" --arsitektur experiments/resnet34_model.py

    Syarat berkasnya:
      - punya fungsi get_model() dan get_transform()
      - get_transform() boleh memakai resolusi apa saja (224, 384, ...); resolusi
        LATIH mengikuti get_transform() secara otomatis, jadi tidak perlu disamakan
        dengan IMG_SIZE
      - get_model() hanya MEMBANGUN arsitektur (weights=None), tidak memuat bobot

    Cara ini dipakai HANYA untuk menguji. Percobaan yang menang baru disalin
    manual ke submission/model.py (lihat experiments/README.md).
    """
    jalur = Path(jalur)
    # Jalur relatif dihitung dari folder kerja. Kalau tidak ketemu di sana,
    # dicari juga di folder tempat train.py berada - supaya tidak gagal hanya
    # karena menjalankannya dari folder lain.
    if not jalur.is_file() and not jalur.is_absolute():
        alternatif = Path(__file__).resolve().parent / jalur
        if alternatif.is_file():
            print(f"[arsitektur] '{jalur}' tidak ada di folder kerja, dipakai: {alternatif}")
            jalur = alternatif
    if not jalur.is_file():
        _dir_exp = Path(__file__).resolve().parent / "experiments"
        _ada = sorted(p.name for p in _dir_exp.glob("*_model.py")) if _dir_exp.is_dir() else []
        raise SystemExit(
            f"[GAGAL] berkas arsitektur tidak ditemukan: {jalur}\n"
            f"  folder kerja : {Path.cwd()}\n"
            f"  yang dicoba  : {jalur.resolve()}\n"
            f"  kandidat yang tersedia: {_ada if _ada else '(tidak ada berkas *_model.py)'}\n"
            "  Contoh: --arsitektur experiments/resnet34_model.py")

    import importlib.util
    nama_modul = "arsitektur_uji_" + jalur.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(nama_modul, jalur)
    modul = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(modul)
    except Exception as e:
        raise SystemExit(
            f"[GAGAL] berkas arsitektur {jalur} gagal dimuat.\n"
            f"  sebab : {type(e).__name__}: {e}")

    for fungsi in ("get_model", "get_transform"):
        if not hasattr(modul, fungsi):
            raise SystemExit(
                f"[GAGAL] {jalur} tidak punya fungsi {fungsi}().\n"
                "  Kedua fungsi itu wajib ada (lihat experiments/README.md).")
    return modul


# ============================================================
# JADWAL LEARNING RATE (warmup + cosine)
# ============================================================
def lr_schedule(epoch: int, warmup: int, base_lr: float, eta_min: float,
                total: int) -> float:
    """LR untuk epoch ke-`epoch` (1-indexed): warmup linear lalu cosine turun."""
    if warmup and epoch <= warmup:
        return base_lr * epoch / float(warmup)
    prog = (epoch - warmup) / float(max(1, total - warmup))
    prog = min(1.0, max(0.0, prog))
    return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * prog))


# ============================================================
# SATU EPOCH LATIH & EVALUASI
# ============================================================
def train_one_epoch(model, loader, loss_fn, opt, device) -> tuple:
    """Latih satu epoch. Return (loss rata-rata, akurasi latih)."""
    model.train()
    total_loss, benar, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # set_to_none=True lebih hemat memori daripada mengisi gradien dengan nol
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
        benar += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), benar / max(1, total)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    """Evaluasi model: loss, akurasi, macro-F1, F1 per kelas, confusion matrix.

    WAJIB memakai @torch.no_grad(). Tanpa itu, PyTorch menyimpan seluruh grafik
    perhitungan untuk SETIAP batch validasi (13 batch pada split 85/15), dan
    memori GPU menumpuk sampai kehabisan (CUDA out of memory).
    """
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        y_pred += logits.argmax(1).cpu().tolist()
        y_true += y.cpu().tolist()

    f1, acc, cm, per_f1 = macro_f1(y_true, y_pred)
    return {"loss": total_loss / max(1, len(y_true)), "acc": acc,
            "macro_f1": f1, "per_class_f1": per_f1, "cm": cm,
            "n": len(y_true)}


# ============================================================
# PROGRAM UTAMA
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Latih ResNet-18 untuk Studio 2")
    ap.add_argument("--data", required=True, help="folder train/ dari dataset")
    ap.add_argument("--backbone", default=None,
                    help="nama model torchvision (resnet18, resnet34, ...) untuk arsitektur bawaan")
    ap.add_argument("--arsitektur", default=None,
                    help="berkas .py di experiments/ berisi get_model()+get_transform() milikmu sendiri")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eta-min", type=float, default=1e-6, help="LR akhir cosine")
    ap.add_argument("--warmup", type=int, default=3, help="epoch warmup LR")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--kfolds", type=int, default=0, help="0 = split tunggal")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--start-epoch", type=int, default=10,
                    help="epoch minimum sebelum early stopping boleh aktif")
    ap.add_argument("--min-delta", type=float, default=0.001,
                    help="kenaikan macro-F1 minimum agar dianggap MAJU (hanya untuk "
                         "early stopping; model terbaik selalu disimpan tanpa ambang)")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="linear probing selama N epoch, lalu buka seluruh backbone")
    ap.add_argument("--scratch", action="store_true",
                    help="latih dari nol tanpa bobot ImageNet (baseline P4)")
    ap.add_argument("--no-class-weight", dest="class_weight", action="store_false",
                    help="matikan pembobotan kelas")
    ap.add_argument("--full-data", action="store_true",
                    help="latih memakai 100 persen data (tanpa validasi) untuk putaran Final")
    ap.add_argument("--val-out", default="./my_val",
                    help="folder validasi lokal untuk diuji evaluate.py")
    ap.add_argument("--no-write-val", dest="write_val", action="store_false")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah gambar (uji cepat)")
    ap.add_argument("--tag", default=None, help="nama folder hasil di runs/")
    ap.add_argument("--out", default=None,
                    help="berkas bobot keluaran; None = runs/<tag>.pth (percobaan), "
                         "atau submission/model.pth kalau --to-submission")
    ap.add_argument("--to-submission", dest="to_submission", action="store_true",
                    help="tulis bobot ke submission/model.pth (untuk putaran Final yang dinilai)")
    a = ap.parse_args()

    # ---------- pilih sumber arsitektur ----------
    # (1) --arsitektur : berkas sendiri di experiments/ (get_model + get_transform)
    # (2) bawaan       : submission/model.py (get_model + get_transform)
    if a.arsitektur:
        modul_arsitektur = muat_arsitektur(a.arsitektur)
        get_model = modul_arsitektur.get_model
        get_transform = modul_arsitektur.get_transform
        print(f"[arsitektur] dari berkas : {a.arsitektur}")
    else:
        modul_arsitektur = None
        print("[arsitektur] bawaan       : submission/model.py")

    # nama backbone untuk pengunduhan bobot ImageNet
    if a.backbone is None:
        a.backbone = getattr(modul_arsitektur, "BACKBONE", "resnet18") if modul_arsitektur else "resnet18"
    if modul_arsitektur is not None and getattr(modul_arsitektur, "BACKBONE", None) is None:
        print(f"[info] {a.arsitektur} tidak menyebut BACKBONE -> bobot ImageNet TIDAK disalin")
        print("       (tambahkan baris  BACKBONE = \"resnet34\"  di berkas itu kalau ingin disalin)")
        a.scratch = True

    device = get_device()
    seed_everything(a.seed)
    workers = a.num_workers
    if workers is None:
        # Windows: DataLoader dengan num_workers > 0 sering menggantung/hang,
        # jadi defaultnya 0 (baca data di proses utama). Di Linux/macOS boleh
        # beberapa worker supaya GPU tidak menunggu.
        workers = 0 if os.name == "nt" else (4 if torch.cuda.is_available() else 0)
    tag = a.tag or f"{a.backbone}_{'full' if a.full_data else ('kf' + str(a.fold) if a.kfolds else 'split')}_seed{a.seed}"
    run_dir = Path("runs") / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- ke mana bobot disimpan ----------
    # Percobaan (bawaan) -> runs/<tag>.pth : tiap percobaan punya berkas sendiri,
    #                       jadi tidak saling menimpa dan submission/ tetap aman.
    # Putaran Final      -> submission/model.pth, hanya kalau --to-submission.
    if a.out:
        out_path = Path(a.out)
        mode_out = "ditunjuk --out"
    elif a.to_submission:
        out_path = Path("./submission/model.pth")
        mode_out = "SUBMISSION (putaran Final)"
    else:
        out_path = run_dir / (tag + ".pth")
        mode_out = "percobaan (submission/ tidak disentuh)"

    print("=" * 68)
    print("  PELATIHAN STUDIO 2 - KLASIFIKASI SAMPAH")
    print("=" * 68)
    print(f"  perangkat      : {device}")
    print(f"  backbone       : {a.backbone}"
          + ("  (TANPA bobot pretrained)" if a.scratch else "  (pretrained ImageNet)"))
    print(f"  epoch maks     : {a.epochs}  |  lr {a.lr}  |  warmup {a.warmup}")
    print(f"  batch          : {a.batch_size}  |  weight decay {a.weight_decay}")
    print(f"  loss           : CrossEntropy (label smoothing {a.label_smoothing}"
          + (", bobot kelas AKTIF)" if a.class_weight else ", bobot kelas mati)"))
    print(f"  seleksi model  : macro-F1 (disimpan sejak epoch 1, tanpa ambang)")
    print(f"  early stopping : patience {a.patience}  |  aktif mulai epoch {a.start_epoch}"
          f"  |  min_delta {a.min_delta}")
    print(f"  hasil          : {run_dir}")
    print(f"  bobot          : {out_path}   [{mode_out}]")

    # ---------- catat konfigurasi (bukti percobaan) ----------
    (run_dir / "config.json").write_text(json.dumps({
        "backbone": a.backbone, "arsitektur": a.arsitektur or "submission/model.py",
        "epochs": a.epochs, "lr": a.lr, "eta_min": a.eta_min, "warmup": a.warmup,
        "weight_decay": a.weight_decay, "label_smoothing": a.label_smoothing,
        "batch_size": a.batch_size, "val_frac": a.val_frac, "kfolds": a.kfolds,
        "fold": a.fold, "seed": a.seed, "patience": a.patience,
        "start_epoch": a.start_epoch, "min_delta": a.min_delta,
        "freeze_epochs": a.freeze_epochs, "scratch": a.scratch,
        "class_weight": a.class_weight, "limit": a.limit, "full_data": a.full_data,
        "data": str(a.data), "out": str(out_path), "tag": tag,
    }, indent=2, default=str))

    # ---------- 1. SPLIT ----------
    per_kelas = limit_per_class(list_dataset(Path(a.data)), a.limit)
    if a.full_data:
        train_items = [it for items in per_kelas for it in items]
        val_items = []
        print(f"\n[split] MODE FULL DATA: latih memakai SEMUA {len(train_items)} gambar")
    elif a.kfolds:
        train_items, val_items = split_kfold(per_kelas, a.kfolds, a.fold, a.seed)
        print(f"\n[split] K-fold {a.kfolds}, fold {a.fold} (berstrata)")
    else:
        train_items, val_items = split_single(per_kelas, a.val_frac, a.seed)
        print(f"\n[split] split tunggal berstrata {a.val_frac:.0%} (seed {a.seed})")
        print(f"  pembagian dihitung ulang di sini (bukan dari berkas) - "
              f"seed {a.seed} dipakai supaya hasilnya bisa diulang persis")

    print(f"  latih {len(train_items)} gambar | validasi {len(val_items)} gambar")
    if not a.full_data:
        n_tr = np.bincount([l for _, l in train_items], minlength=NUM_CLASSES)
        n_va = np.bincount([l for _, l in val_items], minlength=NUM_CLASSES)
        print(f"  {'kelas':<16}{'latih':>8}{'validasi':>10}{'porsi':>8}")
        for i, nama in enumerate(CLASSES):
            porsi = 100.0 * n_va[i] / max(1, n_tr[i] + n_va[i])
            print(f"  {nama:<16}{n_tr[i]:>8}{n_va[i]:>10}{porsi:>7.1f}%")

    if a.write_val and not a.full_data:
        meta = {"seed": a.seed, "val_frac": (1.0 / a.kfolds if a.kfolds else a.val_frac),
                "kfolds": a.kfolds, "fold": a.fold if a.kfolds else 0,
                "n_val": len(val_items), "limit": a.limit,
                "files": sorted(Path(p).name for p, _ in val_items)}
        tulis_my_val(val_items, Path(a.val_out), meta)

    # ---------- 2. DATA LOADER ----------
    # Kontrak: validasi HARUS memakai get_transform() yang sama dengan yang dipakai
    # asisten. Resolusi latih WAJIB mengikuti resolusi get_transform() - kalau tidak,
    # akurasi turun diam-diam. Beberapa backbone memakai resolusi bukan 224
    # (mis. EfficientNet V2-S = 384), jadi augmentasi latih disesuaikan otomatis.
    val_tf = get_transform()
    probe = val_tf(Image.new("RGB", (320, 320)))
    H, W = int(probe.shape[-2]), int(probe.shape[-1])
    if H != W:
        raise SystemExit(f"[GAGAL] get_transform() menghasilkan {H}x{W} (tidak persegi).")
    img_latih = H
    if img_latih != IMG_SIZE:
        print(f"\n[arsitektur] resolusi {img_latih}x{img_latih} (bukan {IMG_SIZE}) "
              "-> augmentasi latih disesuaikan")
    tr_tf = build_train_transform(img_latih)
    tr_ds = PathListDataset(train_items, tr_tf)
    print(f"\n[kontrak] get_transform() -> {H}x{W}  "
          f"(cocok dengan resolusi latih {img_latih}x{img_latih}) OK")

    va_ds = PathListDataset(val_items, val_tf) if val_items else None

    # ---------- PENGAMAN MEMORI: batch otomatis diturunkan bila resolusi besar ----
    # Kebutuhan memori naik ~kuadrat resolusi. Pada 384 (EfficientNet V2-S) batch
    # 32 TIDAK muat di T4 16 GB -> CUDA out of memory di tengah epoch.
    _batas_batch = {224: 32, 256: 24, 288: 16, 320: 12, 384: 8, 448: 6, 512: 4}
    batch_aman = next((max(1, b) for r, b in sorted(_batas_batch.items())
                       if img_latih <= r), 4)
    if a.batch_size > batch_aman:
        print(f"[memori] resolusi {img_latih}x{img_latih} -> batch diturunkan otomatis: "
              f"{a.batch_size} -> {batch_aman}")
        a.batch_size = batch_aman
    else:
        print(f"[memori] batch {a.batch_size} aman untuk resolusi {img_latih}x{img_latih}")

    # drop_last=True: buang batch terakhir yang tidak penuh. Batch kecil
    # (mis. 4 gambar) membuat statistik BatchNorm goyah saat latih.
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=workers,
                       pin_memory=(device.type == "cuda"), drop_last=True)
    va_dl = (DataLoader(va_ds, a.batch_size, shuffle=False, num_workers=workers,
                        pin_memory=(device.type == "cuda")) if va_ds else None)
    # va_dl dibuat dengan batch yang mungkin sudah diturunkan di atas

    # ---------- 3. MODEL ----------
    # Periksa & bersihkan memori GPU DULU, SEBELUM model dibuat. Kalau tidak,
    # sisa memori proses/percobaan sebelumnya membuat model.to(device) gagal
    # dengan pesan CUDA out of memory yang membingungkan.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        bebas, total = torch.cuda.mem_get_info()
        print(f"[memori] GPU bebas {bebas / 1e9:.2f} GB dari {total / 1e9:.2f} GB")
        if bebas < 4e9:
            raise SystemExit(
                f"[GAGAL] memori GPU masih terpakai (hanya {bebas / 1e9:.2f} GB bebas "
                f"dari {total / 1e9:.2f} GB).\n"
                "  Penyebab paling umum: percobaan sebelumnya belum dibebaskan.\n"
                "  SOLUSI: mulai ulang proses/runtime (di Colab: Runtime -> Restart session),\n"
                "  lalu jalankan lagi.")
    model = get_model()
    info_salin = None
    if not a.scratch:
        info_salin = salin_bobot_imagenet(model, a.backbone)
        print(f"[bobot] {a.backbone}: {info_salin['disalin']} tensor disalin dari ImageNet, "
              f"{len(info_salin['dilewati'])} dilewati (termasuk kepala)")
        if info_salin["dilewati"][:5]:
            print(f"        contoh dilewati: {info_salin['dilewati'][:5]}")
        if info_salin["disalin"] < 0.5 * info_salin["total_kunci_model_kita"]:
            print("[PERINGATAN] sangat sedikit tensor yang tersalin - periksa nama backbone!")
    model.to(device)
    print(f"[memori] model sudah di GPU "
          f"({sum(p.numel() for p in model.parameters()):,} parameter)")

    if a.freeze_epochs:
        set_backbone_trainable(model, False)
        print(f"[strategi] linear probing: backbone dibekukan {a.freeze_epochs} epoch, "
              "lalu dibuka penuh (full fine-tuning)")

    # ---------- 4. LOSS + OPTIMIZER ----------
    if a.class_weight and not a.full_data:
        w, counts = class_weight_tensor([l for _, l in train_items], device)
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)
        print("\n[bobot kelas] (N / (C x n_c), dihitung HANYA dari data latih)")
        for i, nama in enumerate(CLASSES):
            print(f"  {nama:<16} {int(counts[i]):>5} gambar  ->  bobot {w[i].item():.3f}")
        bobot_freq = float((counts / counts.sum() * w.cpu().numpy()).sum())
        print(f"  {'rata-rata tertimbang frekuensi':<31} = {bobot_freq:.6f}  (selalu 1,0)")
        print(f"  {'rata-rata biasa (info saja)':<31} = {w.mean().item():.6f}  "
              "(bukan 1,0 karena kelas tidak seimbang)")
    else:
        if a.class_weight and a.full_data:
            train_labels = [l for _, l in train_items]
            w, counts = class_weight_tensor(train_labels, device)
            loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)
            print("\n[bobot kelas] dihitung dari seluruh data latih")
        else:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    latih = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(latih, lr=a.lr, weight_decay=a.weight_decay)
    print(f"\n[optimizer] AdamW, {sum(p.numel() for p in latih):,} parameter dilatih")

    # ---------- 5. LATIH ----------
    epochs = a.epochs
    start_epoch = a.start_epoch
    if start_epoch >= epochs:
        start_epoch = max(1, epochs // 2)
        print(f"[peringatan] --start-epoch >= --epochs -> diturunkan ke {start_epoch}")

    best_f1, best_ep, best_acc, no_improve, tersimpan = -1.0, 0, 0.0, 0, False
    history = []
    print("\n" + "-" * 68)

    for ep in range(1, epochs + 1):
        lr_now = lr_schedule(ep, a.warmup, a.lr, a.eta_min, epochs)

        if a.freeze_epochs and ep == a.freeze_epochs + 1:
            # Backbone dibuka -> WAJIB dimasukkan ke optimizer, kalau tidak
            # parameter baru tidak akan pernah ikut diperbarui.
            baru = set_backbone_trainable(model, True)
            if baru:
                opt.add_param_group({"params": baru, "lr": lr_now,
                                     "weight_decay": a.weight_decay})
            total = sum(p.numel() for g in opt.param_groups for p in g["params"])
            print(f"  [strategi] epoch {ep}: seluruh backbone dibuka -> "
                  f"{len(baru)} tensor masuk optimizer "
                  f"(total {total:,} parameter dilatih)")

        for g in opt.param_groups:
            g["lr"] = lr_now

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, tr_dl, loss_fn, opt, device)
        row = {"epoch": ep, "lr": lr_now, "train_loss": tr_loss, "train_acc": tr_acc}

        if va_dl is None:
            # Mode full-data: tidak ada validasi, bobot disimpan tiap epoch
            # sehingga berkas akhir = epoch terakhir.
            torch.save(model.state_dict(), out_path)
            tersimpan = True
            best_ep = ep
            print(f"  epoch {ep:2d}/{epochs} | train_loss {tr_loss:.4f} "
                  f"acc {tr_acc:.4f} | lr {lr_now:.2e} | {time.time()-t0:.1f}s")
        else:
            m = evaluate(model, va_dl, loss_fn, device)
            row.update({"val_loss": m["loss"], "val_acc": m["acc"],
                        "val_macro_f1": m["macro_f1"]})
            tanda = ""
            # SIMPAN model terbaik sejak epoch 1 (tanpa min_delta).
            # Versi lama baru menyimpan mulai epoch `start_epoch` dan memakai
            # ambang min_delta, sehingga model terbaik di epoch awal terbuang.
            if m["macro_f1"] > best_f1:
                best_f1, best_ep, best_acc = m["macro_f1"], ep, m["acc"]
                no_improve = 0
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
                tersimpan = True
                tanda = "  <-- TERBAIK, disimpan"
            elif ep >= start_epoch:
                # Early stopping baru dihitung setelah pemanasan epoch awal.
                # min_delta di sini berarti "kenaikan minimum agar dianggap maju".
                if m["macro_f1"] > best_f1 + a.min_delta:
                    no_improve = 0
                else:
                    no_improve += 1
            print(f"  epoch {ep:2d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.4f} | "
                  f"val {m['loss']:.4f}/{m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
                  f"lr {lr_now:.2e} | {time.time()-t0:.1f}s{tanda}")

        history.append(row)

        if va_dl is not None and ep >= start_epoch and no_improve >= a.patience:
            print(f"  -> EARLY STOPPING di epoch {ep} "
                  f"(macro-F1 tidak membaik {a.patience} epoch)")
            break

    print("-" * 68)

    # ---------- 6. LAPORAN AKHIR ----------
    laporan = {"backbone": a.backbone, "pretrained": not a.scratch,
               "arsitektur": (a.arsitektur or "submission/model.py"),
               "epochs_dijalankan": len(history), "tag": tag}
    if info_salin:
        laporan["salin_bobot"] = {"disalin": info_salin["disalin"],
                                  "dilewati": len(info_salin["dilewati"])}

    if va_dl is not None:
        if not tersimpan:
            print("[peringatan] tidak ada epoch yang tersimpan -> menyimpan epoch terakhir")
            torch.save(model.state_dict(), out_path)
            best_ep = len(history)
        model.load_state_dict(torch.load(out_path, map_location=device))
        akhir = evaluate(model, va_dl, loss_fn, device)
        print(f"\nHASIL TERBAIK (epoch {best_ep})")
        print(f"  Macro-F1 : {akhir['macro_f1']:.4f}   <-- metrik penilaian")
        print(f"  Akurasi  : {akhir['acc']:.4f}")
        print("  F1 per kelas:")
        for i, nama in enumerate(CLASSES):
            print(f"    {nama:<16} {akhir['per_class_f1'][i]:.4f}")
        print("  Confusion matrix (baris = asli, kolom = prediksi):")
        for i, nama in enumerate(CLASSES):
            print(f"    {nama:<16} {akhir['cm'][i].tolist()}")
        laporan.update({"best_epoch": best_ep,
                        "val_macro_f1": akhir["macro_f1"],
                        "val_acc": akhir["acc"],
                        "per_class_f1": dict(zip(CLASSES, akhir["per_class_f1"])),
                        "confusion_matrix": akhir["cm"].tolist()})
    else:
        print(f"\nMODE FULL DATA selesai. Bobot akhir -> {out_path}")
        print("  Tidak ada skor validasi (semua data dipakai latih).")
        laporan.update({"full_data": True, "epochs_dijalankan": len(history)})

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(laporan, indent=2))

    print(f"\nBobot   : {out_path}")
    print(f"Bukti   : {run_dir}/(config|history|metrics).json")
    if va_dl is not None:
        print("\nLANGKAH WAJIB BERIKUTNYA - uji dengan skrip resmi milik asisten:")
        print(f"  python evaluate.py --submission ./submission --data {a.val_out}")


if __name__ == "__main__":
    main()
