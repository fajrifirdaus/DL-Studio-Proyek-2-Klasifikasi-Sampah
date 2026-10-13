# Studio 2 — Kompetisi Klasifikasi Sampah

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

**Nama kelompok:** `<15/DUO FF>` — *sesuaikan dengan nama yang didaftarkan ke asisten*

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Fadhilah Akbar | 123140003 | *(diisi kelompok)* |
| 2  | Muhammad Fajri Firdaus  | 123140050 | *(diisi kelompok)* |

Kelompok beranggotakan maksimal 3 orang. Nomor 1–2 di atas sudah sesuai jumlah anggota
yang didaftarkan. **NIM wajib tercantum** — README tanpa NIM tidak dapat dievaluasi.

## Ringkasan Pendekatan

| Butir | Isi |
|---|---|
| **Arsitektur / backbone** | **ResNet-18** pretrained ImageNet (`torchvision`, bobot `IMAGENET1K_V1`) |
| **Strategi pelatihan** | Full fine-tuning (seluruh layer dilatih) setelah 3 epoch *linear probing*; AdamW, warmup linear + cosine decay, 30 epoch maksimal, early stopping |
| **Pemilihan model** | Berdasarkan **macro-F1 validasi** (bukan akurasi), sesuai metrik penilaian |
| **Augmentasi** | RandomResizedCrop(224, 0.60–1.00), horizontal flip, ColorJitter, RandomErasing |
| **Penanganan kelas tidak seimbang** | Bobot kelas `N / (C × n_c)` pada CrossEntropyLoss, dihitung hanya dari data latih |
| **Label noise** | Label smoothing 0,1 |
| **Data tambahan** | **Tidak ada** — hanya train set dari starter kit (sesuai ketentuan, dataset tidak diunggah ulang) |
| **Ukuran input & preprocessing** | `Resize(256)` → `CenterCrop(224)` → `ToTensor` → normalisasi ImageNet (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`) |
| **Paket** | Hanya paket dasar: `torch`, `torchvision`, `pillow`, `numpy` — **tanpa timm** |
| **Ukuran `model.pth`** | ±44,7 MB (jauh di bawah batas 100 MB GitHub dan 500 MB aturan) |

### Mencoba Backbone Lain (ResNet-34, ResNet-50, …)

Boleh dicoba, tapi **jangan menempel langsung ke `submission/`**. Alurnya:

| # | Langkah |
|---|---|
| 1 | Buat berkas kandidat di `experiments/` (contoh: `resnet34_model.py`) |
| 2 | Di notebook, sel KONFIGURASI: `CFG["arsitektur"] = "experiments/resnet34_model.py"`, `CFG["backbone"] = "resnet34"`, `CFG["tag"] = "coba_resnet34"` |
| 3 | Jalankan ulang sel data loader & LATIH — bobotnya masuk `runs/<tag>.pth`, `submission/` **tidak tersentuh** |
| 4 | Bandingkan angka di `runs/<tag>/metrics.json` — atau jalankan sel **20 (Bandingkan Semua Percobaan)** untuk tabel otomatis |
| 5 | Kalau menang → **salin manual** ke `submission/model.py`, lalu set `CFG["tulis_submission"] = True` (notebook) atau tambah `--to-submission` (laptop), dan latih ulang |

Di laptop sama saja:

```bash
# ResNet-34 (224, 83,3 MB)
python train.py --data "<...>/train" --arsitektur experiments/resnet34_model.py \
                --backbone resnet34 --tag coba_resnet34

# EfficientNet V2-S (384, 82,7 MB) - latih ~3,5x lebih lama
# batch-nya diturunkan OTOMATIS jadi 8 (resolusi 384 tidak muat di batch 32)
python train.py --data "<...>/train" --arsitektur experiments/efficientnet_v2s_model.py \
                --backbone efficientnet_v2_s --tag coba_v2s   # batch otomatis jadi 8
```

> **Kenapa manual, bukan otomatis?** Supaya tidak ada yang bisa menimpa submission secara
> tak sengaja. `submission/model.py` dan `submission/model.pth` wajib **berpasangan** —
> kalau `model.py` jadi ResNet-34 tetapi `model.pth` masih ResNet-18, evaluasi asisten gagal.

> **Jujur:** ganti backbone belum tentu menaikkan skor. Baseline resmi 0,92 dan ruang
> perbaikan terbesar ada di **cara melatih** — itu yang sudah dikerjakan `train.py`.
> Backbone lebih besar = latih 1,5–2× lebih lama, `model.pth` ±85 MB.
> **Memori:** batch diturunkan otomatis mengikuti resolusi
> (224→32, 256→24, 288→16, 320→12, 384→8, 448→6, 512→4), jadi 384 aman di T4 16 GB.

## Kenapa ResNet-18, bukan backbone yang lebih besar?

1. **Baseline resmi kelas ini adalah ResNet-18 pretrained dengan 8 epoch → macro-F1 ≈ 0,92.**
   Artinya angka itu adalah *lantai*, bukan atap: baseline belum dioptimasi sama sekali
   (tanpa bobot kelas, tanpa early stopping, seleksi memakai akurasi, augmentasi minim).
2. **Validation set tersembunyi hanya ±740 gambar (≈148 per kelas).** Dengan jumlah itu,
   simpangan baku macro-F1 sekitar ±1%, sehingga selisih antar-backbone yang diperkirakan
   1–2% **tidak dapat dibedakan dari keberuntungan**. Mengganti backbone ke model lebih besar
   menambah risiko (transform berbeda, memori GPU) untuk keuntungan yang belum terbukti.
3. **Ruang perbaikan terbesar ada di cara melatih, bukan di arsitektur.** Karena itu fokus
   kelompok ini adalah memperbaiki pipeline pelatihan, dengan catatan pada bagian
   "Celah yang diperbaiki".

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | *(belum)* | *(belum)* | — |
| Preview 2 (21 Okt) | *(belum)* | *(belum)* | — |
| Final (25 Okt)     | *(belum)* | **tidak ada** (dilatih ulang 100% data) | — |

Sumber angka: `runs/<tag>/metrics.json` dan keluaran `evaluate.py`.
Angka kolom asisten diisi setelah pengumuman di Google Sheets skor kelompok.

## Celah yang Diperbaiki dari `train_baseline.py`

`train_baseline.py` dari starter kit tetap disimpan sebagai acuan. `train.py` adalah versi
kelompok dengan 10 perbaikan berikut.

> **Yang paling berpengaruh nomor 3 dan 4** (seleksi macro-F1 + pembobotan kelas) — keduanya
> langsung menyasar metrik penilaian. Nomor 1 hanya soal pembulatan (selisih 3 gambar).

| # | Celah pada baseline | Perbaikan di `train.py` |
|---|---|---|
| 1 | Pembulatan `int(n×0,15)` per kelas | `max(1, round(n×0,15))` → tiap kelas tepat 15% (413 vs 410 gambar) |
| 2 | `my_val/` lama dipakai ulang tanpa dicek (split kedaluwarsa) | Penanda `.split.json`; split dibangun ulang bila seed/frac berubah |
| 3 | Model terbaik dipilih berdasarkan **akurasi** | Dipilih berdasarkan **macro-F1** (metrik penilaian asisten) |
| 4 | Tanpa pembobotan kelas → kelas minoritas kalah | **Bobot kelas** `N/(C×n_c)`, hanya dari data latih |
| 5 | Tanpa early stopping | Early stopping (patience 8, aktif mulai epoch 10) + model terbaik disimpan **sejak epoch 1** |
| 6 | Augmentasi belum mengikuti resep terbaik yang terbukti | Resep disamakan: crop 0.60–1.00, flip, ColorJitter 0.3/0.3/0.3/0.05, RandomErasing p=0.25 (rotasi & crop sempit dibuang) |
| 7 | Tidak ada pemeriksaan kontrak transform | Resolusi `get_transform()` dicek otomatis terhadap resolusi latih (mismatch = gagal cepat, bukan akurasi turun diam-diam) |
| 8 | Salin bobot hardcode awalan `fc.` (khusus ResNet) | Salin berbasis kecocokan bentuk tensor + dilaporkan jumlahnya |
| 9 | Tidak pernah diuji lewat `evaluate.py` sebelum tenggat | Pelatihan memakai `get_model()`/`get_transform()` dari `submission/model.py`, jadi mismatch kontrak terdeteksi lebih awal |
| 10 | Tidak mendukung strategi fine-tuning yang diminta RPS | Dukungan *linear probing* → *full fine-tuning*, 5-fold CV, dan pelatihan ulang 100% data |
| 11 | `evaluate()` tanpa `@torch.no_grad()` → grafik perhitungan menumpuk di VRAM (CUDA out of memory) | `@torch.no_grad()` + `zero_grad(set_to_none=True)` + batch otomatis + `empty_cache()` |

**Catatan kejujuran:** nilai harapan perbaikan (mis. "+1–3%") tidak dicantumkan di sini karena
belum diukur. Yang terukur baru baseline resmi 0,92 pada validation set tersembunyi.

## Cara Reproduksi

Jalankan dari **root repo**. Urutan berkas: `train_baseline.py` (acuan dosen) → `train.py` (versi kelompok).

### 1. Latih

```bash
# Pasang dependensi
pip install -r requirements.txt

# Uji cepat ±1 menit — pastikan skrip jalan sebelum latih penuh
python train.py --data "<...>/Dataset Sampah Studio 2/train" --limit 300 --epochs 2

# Pelatihan utama (holdout beku, seleksi macro-F1)
python train.py --data "<...>/Dataset Sampah Studio 2/train"

# WAJIB: uji dengan skrip evaluasi resmi milik asisten
python evaluate.py --submission ./submission --data ./my_val

# Eksperimen pendukung (diminta RPS: freezing vs linear probing vs full fine-tuning)
python train.py --data "<...>/train" --freeze-epochs 5    # linear probing lalu full fine-tuning
python train.py --data "<...>/train" --scratch            # CNN from scratch, tanpa bobot ImageNet
python train.py --data "<...>/train" --no-class-weight    # ablasi bobot kelas

# 5-fold cross-validation — memilih konfigurasi tanpa "mengintip" validasi
python train.py --data "<...>/train" --kfolds 5 --fold 1   # ulangi --fold 2..5

# Putaran FINAL: latih ulang di 100% data (tidak ada data sisa)
python train.py --data "<...>/train" --full-data --epochs 18
```

Setiap run membuat folder `runs/<tag>/` berisi `config.json`, `history.json`, dan
`metrics.json` sebagai bukti eksperimen.

### 2. Promosikan kandidat dengan aman

Kandidat dilatih ke `runs/`, diuji dulu, baru ditempel ke `submission/` — supaya
`submission/` selalu dalam keadaan bisa dinilai.

```bash
# Latih kandidat -> bobotnya OTOMATIS ke runs/kandidat_a.pth (submission/ aman)
python train.py --data "<...>/train" --tag kandidat_a

# Uji dulu, baru promosikan
python experiments/promote.py --candidate runs/kandidat_a.pth --dry-run
python experiments/promote.py --candidate runs/kandidat_a.pth --yes
```


### Perubahan yang mengubah `submission/model.py`

Bila resolusi latih diubah, `get_transform()` **wajib** diubah berpasangan:

| Resolusi latih | `get_transform()` |
|---|---|
| 224 (dipakai sekarang) | `Resize(256)` + `CenterCrop(224)` |
| 288 | `Resize(320)` + `CenterCrop(288)` |

Melatih pada satu resolusi tetapi mengevaluasi pada resolusi lain membuat akurasi
turun **tanpa pesan error**. `train.py` memeriksa hal ini secara otomatis.

## Catatan Teknis Penting

- **Urutan kelas wajib alfabetis** (sama dengan `ImageFolder` dan `evaluate.py`):
  `0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam`
- **`model.py` hanya membangun arsitektur**: `weights=None`, tanpa mengunduh bobot,
  tanpa memuat `model.pth` (yang memuat adalah `evaluate.py`).
- **Batas teknis:** `model.pth` ≤ 500 MB dan evaluasi seluruh validation set ≤ 15 menit.
  `model.pth` kelompok ini ±44,7 MB, jadi tidak perlu GitHub Release / LFS.
- **Kompatibilitas `strict=True`:** karena model tidak diubah dan `model.pth` dihasilkan
  oleh `get_model()` dari `model.py` yang sama, semua nama kunci `state_dict` cocok.

## Distribusi Dataset yang Dipakai

| Kelas | Jumlah gambar | Bobot kelas yang dipakai |
|---|---:|---:|
| botol_plastik | 523 | 1,053 |
| gelas_plastik | 521 | 1,057 |
| kertas_kardus | 365 | 1,508 |
| saset_kemasan | 673 | 0,818 |
| styrofoam | 671 | 0,821 |
| **Total** | **2.753** | rata-rata *tertimbang frekuensi* 1,000 |

Rasio kelas terbesar : terkecil = **1,84×**. Kelas `kertas_kardus` menjadi perhatian utama
karena jumlahnya paling sedikit.

> **Catatan rumus:** bobotnya `N / (C × n_c)`. Yang bernilai **1,0** adalah rata-rata
> **tertimbang frekuensi** (`Σ (n_c/N)·w_c = 1,000`). Rata-rata biasa antar kelas adalah
> **1,051**, bukan 1,0 — angka ini memang bukan 1,0 karena kelasnya tidak seimbang, jadi
> jangan dipakai sebagai penanda kesalahan.

> **Catatan jumlah data:** salinan lokal berisi **2.753** gambar (per kelas: botol 523,
> gelas 521, kertas 365, saset 673, styrofoam 671). Halaman resmi menyebut ±2.900 —
> selisihnya dibahas di bagian **Catatan `train_sources.csv`** di bawah; dataset ini
> sudah yang paling baru dan **tidak perlu diunduh ulang**.

## Struktur Repositori

```
.
├── README.md                  # berkas ini (nama, NIM, pendekatan, hasil, cara reproduksi)
├── requirements.txt           # dependensi untuk melatih
├── evaluate.py                # salinan skrip evaluasi resmi (tidak diubah)
├── train.py                   # pipeline pelatihan versi kelompok (memperbaiki 10 celah)
├── train_baseline.py          # baseline resmi dari starter kit (acuan, tidak diubah)
├── notebooks/train_colab.ipynb  # SATU-SATUNYA tempat melatih (Colab, GPU gratis)
├── notebooks/
│   ├── train_colab.ipynb      # runner Colab: ambil ZIP dari Drive-mu, latih, unduh hasil
│   └── README.md              # cara pakai notebook + isi ZIP yang dibutuhkan
├── experiments/               # ruang kerja KANDIDAT (belum tentu bagus)
│   ├── resnet34_model.py      # kandidat ResNet-34 (224, 83,3 MB)
│   ├── efficientnet_v2s_model.py  # kandidat EfficientNet V2-S (384, 82,7 MB)
│   ├── promote.py             # uji kandidat -> promosikan ke submission/ hanya kalau menang
│   └── README.md              # cara mencoba backbone lain + cara salin ke submission/
├── submission/
│   ├── model.py               # get_model() & get_transform()
│   ├── model.pth              # state_dict model terbaik
│   └── requirements.txt       # paket saat evaluasi (paket dasar, tanpa timm)
└── runs/<tag>/                # bukti eksperimen: config, history, metrics (JSON)
```

`my_val/`, `my_train/`, dan dataset tidak di-commit (lihat `.gitignore`) — sesuai ketentuan
agar dataset tidak diunggah ulang ke repositori publik. `train_sources.csv` juga tidak
di-commit karena memuat catatan sumber setiap gambar.

`submission/model.py` dan `evaluate.py` **identik byte-per-byte** dengan starter kit resmi
(sudah diperiksa terhadap berkas di halaman Studio 2). Jadi submission ini memakai
arsitektur contoh yang disediakan dosen — yang kelompok kembangkan adalah **pipeline
pelatihannya** (`train.py`).

## Ke Mana Bobot (`.pth`) Disimpan?

| Mode | Perintah | Lokasi bobot |
|---|---|---|
| **Percobaan** (bawaan) | `python train.py --data "<...>/train" --tag coba_a` | `runs/coba_a.pth` |
| **Putaran Final** | tambah `--to-submission` | `submission/model.pth` |

Karena bawaannya menulis ke `runs/<tag>.pth`, **tiap percobaan punya berkas bobotnya
sendiri** — tidak saling menimpa, dan `submission/` tidak pernah rusak karena percobaan.

Di notebook, setaranya: `CFG["tulis_submission"] = False` (percobaan, default) atau
`True` (Final).

> Isi `runs/<tag>/` tiap percobaan: `config.json`, `history.json`, `metrics.json`
> (angka macro-F1, akurasi, F1 per kelas). Itulah bukti untuk tabel di README.

## Pembagian Data (Train / Validasi) — Dihitung Saat Melatih

Tidak ada berkas daftar split yang di-commit. Pembagiannya **dihitung saat melatih**,
memakai **seed tetap** (`seed = 42`) dan **berstrata** per kelas:

| | |
|---|---|
| **Latih** | 2.340 gambar (85%) |
| **Validasi** | 413 gambar (15%) — botol 78 · gelas 78 · kertas 55 · saset 101 · styrofoam 101 |

Karena algoritmanya sama (Python `random` dengan seed tetap), pembagiannya **identik**
setiap kali dijalankan ulang — di Colab maupun di laptop — tanpa perlu berkas terpisah.

> **Catatan jujur:** angka validasi kita **tidak** bisa dibandingkan langsung dengan
> angka tim lain, karena daftar gambar ujiannya berbeda (walaupun jumlah dan porsinya
> sama). Pembanding yang paling sah adalah kolom hasil di Google Sheets asisten.
> Yang membuat perbandingan antar-run kita sendiri sah adalah seed tetap ini.

K-fold (opsional) juga dihitung saat melatih: `python train.py --kfolds 5 --fold 1`.

## Catatan `train_sources.csv`

Dataset dilengkapi `train_sources.csv` (kolom: `file`, `label`, `image_url`, `page_url`,
`title`). Isinya **2.964 baris**, sedangkan folder `train/` berisi **2.753 gambar** —
selisih 211 baris. Sudah diperiksa penyebabnya:

| Pemeriksaan | Hasil |
|---|---|
| Nama file di CSV berulang? | Tidak — 2.964 nama unik |
| Gambar di disk yang tidak tercatat di CSV | **0** (semua 2.753 tercatat) |
| File tercatat CSV yang tidak ada di disk | 211 |
| Selisih per kelas | botol −18 · gelas −51 · kertas −30 · saset −85 · styrofoam −27 |

Artinya CSV adalah catatan **saat pengumpulan**, sedangkan folder `train/` adalah hasil
**setelah pembersihan** (deduplikasi + penyaringan otomatis + pemeriksaan manual oleh
asisten). 211 berkas yang tercatat di CSV memang dibuang pada tahap pembersihan — sesuai
penjelasan halaman Studio 2 ("dikumpulkan dari web lalu dibersihkan"). Karena itu
**2.753 adalah jumlah yang benar dan terkini**, bukan dataset yang terpotong.

Perbandingan dengan angka di halaman resmi (±2.900): selisih total sekitar 5% dan proporsi
antar kelas tetap sama (saset dan styrofoam terbanyak, kertas_kardus paling sedikit).
Pelatihan tetap memakai seluruh 2.753 gambar yang tersedia.

## Sumber Daya

- Halaman resmi Studio 2: https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/
- Materi Pertemuan 4–5 (CNN from scratch, variasi arsitektur) dan Pertemuan 6 (fine-tuning ImageNet)
- Dataset: train set dari Google Drive resmi mata kuliah (kumpulan citra dari internet,
  untuk keperluan pendidikan non-komersial)
