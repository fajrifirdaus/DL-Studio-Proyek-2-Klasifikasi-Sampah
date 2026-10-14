# Studio 2 — Kompetisi Klasifikasi Sampah

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

**Nama kelompok:** `<15/DUO FF>`

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Fadhilah Akbar | 123140003 | `train.py` (perbaikan celah baseline: seleksi macro-F1, bobot kelas, early stopping), `notebooks/train_colab.ipynb` (alur Colab), `experiments/` (kandidat ResNet-34 & EfficientNet V2-S, `promote.py`), pelatihan kedua kandidat | 
| 2  | Muhammad Fajri Firdaus  | 123140050 | `submission/` (EfficientNet V2-S @384 + bobot terlatih), `evaluate.py` (uji kontrak submission), `README.md` + dokumentasi subfolder, verifikasi bebas kontaminasi split validasi (diagnostik kebocoran & McNemar) |

Repo: https://github.com/fajrifirdaus/DL-Studio-Proyek-2-Klasifikasi-Sampah

## Ringkasan Pendekatan

| Butir | Isi |
|---|---|
| **Arsitektur / backbone** | **EfficientNet V2-S** pretrained ImageNet (`torchvision.efficientnet_v2_s`) |
| **Strategi pelatihan** | Full fine-tuning (seluruh layer dilatih); AdamW, warmup linear + cosine decay, maksimal 30 epoch, early stopping, model terbaik dipilih berdasarkan **macro-F1 validasi** |
| **Augmentasi** | RandomResizedCrop(384, 0.60–1.00), horizontal flip, ColorJitter(0.3/0.3/0.3/0.05), RandomErasing p=0.25 + normalisasi ImageNet |
| **Penanganan kelas tidak seimbang** | Bobot kelas `N / (C × n_c)` pada CrossEntropyLoss, dihitung hanya dari data latih |
| **Label noise** | Label smoothing 0,1 |
| **Data tambahan** | **Tidak ada** — hanya train set dari starter kit; dataset tidak diunggah ulang |
| **Ukuran input & preprocessing** | `Resize(384)` → `CenterCrop(384)` → `ToTensor` → normalisasi ImageNet (`mean [0.485,0.456,0.406]`, `std [0.229,0.224,0.225]`, resolusi asli EfficientNet V2-S) |
| **Paket** | `torch`, `torchvision`, `pillow`, `numpy` — **tanpa timm** |
| **Ukuran `submission/model.pth`** | 81,7 MB (batas GitHub 100 MB; batas aturan 500 MB) |
| **Waktu evaluasi** | 413 gambar = **6,9 detik** pada GPU lokal (setara ±12 detik untuk ±740 gambar) — jauh di bawah batas aturan 15 menit |

## Hasil

Kolom "lokal" = macro-F1 epoch terbaik pada split validasi lokal sendiri (413 gambar, seed 42). Kolom "asisten" = skor yang diumumkan pada validation set tersembunyi.

| Putaran | Commit | Backbone | Macro-F1 lokal | Akurasi lokal | Macro-F1 asisten |
|---|---|---|---|---|---|
| Preview 1 (14 Okt) | [commit `submission/`](https://github.com/fajrifirdaus/DL-Studio-Proyek-2-Klasifikasi-Sampah/commits/main/submission/model.py) | EfficientNet V2-S | 0,9618 | 0,9661 | *(menunggu pengumuman)* |
| Preview 2 (21 Okt) | *(belum)* | — | — | — | — |
| Final (25 Okt) | *(belum)* | — | — | — | — |

Tautan pada kolom **Commit** menunjuk riwayat commit `submission/model.py` — berisi hash commit
Preview 1 beserta seluruh perubahan sesudahnya. Hash pastinya bisa dilihat dengan
`git log -1 --format=%H -- submission/model.py`.

Sumber angka: `runs/<tag>/metrics.json` (berkas JSON-nya di-commit sebagai bukti) dan keluaran resmi
`python evaluate.py --submission ./submission --data ./my_val` (diulang persis seperti asisten).

### Perbandingan backbone yang diuji

Dua kandidat dilatih pada **split validasi yang sama** (413 gambar, seed 42, batch otomatis: 32 untuk 224 px, 8 untuk 384 px), seleksi berdasarkan macro-F1.

| Kandidat | Resolusi | Epoch terbaik | Macro-F1 | Akurasi | F1 kertas_kardus |
|---|---|---|---|---|---|
| ResNet-34 | 224 | 22/30 | 0,9408 | 94,19% | 0,925 |
| **EfficientNet V2-S** | **384** | **16/24** | **0,9618** | **96,61%** | 0,925 |

EfficientNet V2-S dipilih (+2,1 poin macro-F1) dan dipasang ke `submission/`.

> **Catatan jujur 1 — angka lokal overstate.** Kedua angka di atas adalah epoch terbaik **pada himpunan yang dipakai memilih epoch**, jadi bias optimistis. Pada uji terpisah yang bebas kontaminasi (328 gambar validasi yang tidak pernah dipakai seleksi), selisihnya +2,7 pp (96,04% vs 93,29%). Uji McNemar pada 413 gambar: 14 vs 4 gambar, p = 0,031 — beda nyata di split ini, tetapi masih di bawah ambang noise ±4,8 pp yang berlaku pada val sekecil ini. Karena itu angka di tabel ini **dilaporkan sebagai "epoch terbaik pada val lokal"**, bukan perkiraan skor validation set tersembunyi; pembanding yang sah adalah kolom asisten.
> **Catatan jujur 2 — `--kfolds 5` tidak untuk skor.** Pada pipeline ini, run K-fold selalu memakai seluruh data di luar `my_val/` sebagai data latih, sehingga lipatan validasinya sebagian besar sudah pernah dilihat model (terukur 468 dari 553 gambar). K-fold di repo ini berguna untuk uji sensitivitas, **bukan** sebagai estimasi performa jujur. Estimasi jujur diperoleh dengan melatih ulang dari nol pada data latih lipatan itu.

## Celah yang Diperbaiki dari `train_baseline.py`

`train_baseline.py` dari starter kit tetap disimpan sebagai acuan. `train.py` adalah versi kelompok; yang paling berpengaruh nomor 3 dan 4 karena langsung menyasar metrik penilaian.

| # | Celah pada baseline | Perbaikan di `train.py` |
|---|---|---|
| 1 | Pembulatan `int(n×0,15)` per kelas | `max(1, round(n×0,15))` → tiap kelas tepat 15% (413 gambar) |
| 2 | `my_val/` lama dipakai ulang tanpa dicek | Penanda `.split.json`; split dibangun ulang bila seed/frac berubah |
| 3 | Model terbaik dipilih berdasarkan **akurasi** | Dipilih berdasarkan **macro-F1** (metrik penilaian asisten) |
| 4 | Tanpa pembobotan kelas → kelas minoritas kalah | Bobot kelas `N/(C×n_c)`, dihitung hanya dari data latih |
| 5 | Tanpa early stopping | Early stopping (patience 8, aktif mulai epoch 10) + model terbaik disimpan sejak epoch 1 |
| 6 | Augmentasi belum mengikuti resep terbaik | Crop 0.60–1.00, flip, ColorJitter 0.3/0.3/0.3/0.05, RandomErasing p=0.25 (rotasi dibuang) |
| 7 | Tidak ada pemeriksaan kontrak transform | Resolusi `get_transform()` dicocokkan otomatis dengan resolusi latih |
| 8 | Salin bobot hardcode awalan `fc.` | Salin berbasis kecocokan bentuk tensor + dilaporkan jumlahnya |
| 9 | Tidak diuji lewat `evaluate.py` sebelum tenggat | Pelatihan memakai `get_model()`/`get_transform()` dari berkas arsitektur yang sama dengan submission |
| 10 | Tak mendukung strategi fine-tuning di RPS | Linear probing → full fine-tuning, 5-fold CV, dan pelatihan ulang 100% data |
| 11 | `evaluate()` tanpa `@torch.no_grad()` → CUDA OOM | `@torch.no_grad()` + `zero_grad(set_to_none=True)` + batch otomatis + `empty_cache()` |

## Cara Reproduksi

Dijalankan dari **root repo**. Dataset train set tidak di-commit; arahkan `--data` ke folder `train/` hasil unduhan resmi.

```bash
pip install -r requirements.txt

# Uji cepat (±1 menit) — pastikan skrip jalan sebelum latih penuh
python train.py --data "<...>/Dataset Sampah Studio 2/train" --limit 300 --epochs 2

# Latih kandidat (bobot otomatis ke runs/<tag>/<tag>.pth, submission/ tidak tersentuh)
python train.py --data "<...>/train" --arsitektur experiments/resnet34_model.py \
                --backbone resnet34 --tag coba_resnet34
python train.py --data "<...>/train" --arsitektur experiments/efficientnet_v2s_model.py \
                --backbone efficientnet_v2_s --batch-size 8 --tag coba_v2s

# Uji kandidat, lalu promosikan ke submission/ hanya kalau menang
python experiments/promote.py --candidate runs/coba_v2s/coba_v2s.pth \
       --arsitektur experiments/efficientnet_v2s_model.py --dry-run
python experiments/promote.py --candidate runs/coba_v2s/coba_v2s.pth \
       --arsitektur experiments/efficientnet_v2s_model.py --yes

# WAJIB: uji dengan skrip evaluasi resmi milik asisten
python evaluate.py --submission ./submission --data ./my_val
```

Eksperimen pendukung (RPS): `--freeze-epochs 5` (linear probing → full fine-tuning),
`--scratch` (tanpa bobot ImageNet), `--no-class-weight` (ablasi bobot kelas),
`--kfolds 5 --fold 1..5` (lihat Catatan jujur 2). Putaran Final: `--full-data --epochs 18`
melatih ulang di 100% data. Setiap run menulis `runs/<tag>/{config,history,metrics}.json`.

### Perubahan resolusi = perubahan `submission/model.py`

| Resolusi latih | `get_transform()` di `model.py` |
|---|---|
| 224 (ResNet, MobileNet, DenseNet, ConvNeXt) | `Resize(256)` + `CenterCrop(224)` |
| **384 (dipakai sekarang, EfficientNet V2-S)** | `Resize(384)` + `CenterCrop(384)` |

Melatih pada satu resolusi tetapi mengevaluasi pada resolusi lain menurunkan akurasi
**tanpa pesan error** — karena itu `model.py` dan `model.pth` wajib dari eksperimen yang sama.

## Struktur Repositori

```
.
├── README.md                     # berkas ini
├── requirements.txt              # dependensi melatih
├── evaluate.py                   # salinan skrip evaluasi resmi (tidak diubah)
├── train.py                      # pipeline pelatihan versi kelompok
├── train_baseline.py             # baseline resmi starter kit (acuan, tidak diubah)
├── notebooks/train_colab.ipynb   # runner pelatihan di Colab
├── experiments/
│   ├── resnet34_model.py         # kandidat ResNet-34 (224)
│   ├── efficientnet_v2s_model.py # kandidat EfficientNet V2-S (384) -> TERPILIH
│   ├── promote.py                # uji kandidat -> promosi ke submission/ kalau menang
│   └── README.md
├── submission/
│   ├── model.py                  # get_model() EfficientNet V2-S + get_transform() 384
│   ├── model.pth                 # state_dict model terbaik (81,7 MB)
│   └── requirements.txt
└── runs/<tag>/                   # bukti eksperimen: config, history, metrics (JSON)
```

`my_val/`, `train_sources.csv`, dan dataset tidak di-commit (lihat `.gitignore`) sesuai
ketentuan agar dataset tidak diunggah ulang ke repositori publik. Riwayat promosi ada di
Berkas bukti angka tiap percobaan (`runs/<tag>/{config,history,metrics}.json` dan
`kurva_pelatihan.png`) serta `runs/_arsip/riwayat.jsonl` IKUT di-commit; yang diabaikan hanya
bobot `*.pth` di `runs/` dan seluruh isi `runs/_arsip/<cap-waktu>/`.

## Distribusi Dataset yang Dipakai

| Kelas | Jumlah gambar | Bobot kelas `N/(C×n_c)` |
|---|---:|---:|
| botol_plastik | 523 | 1,053 |
| gelas_plastik | 521 | 1,057 |
| kertas_kardus | 365 | 1,508 |
| saset_kemasan | 673 | 0,818 |
| styrofoam | 671 | 0,821 |
| **Total** | **2.753** | rata-rata tertimbang frekuensi 1,000 |

Rasio kelas terbesar : terkecil = **1,84×**. `kertas_kardus` paling sedikit dan menjadi
kelas dengan F1 terendah pada kedua kandidat (0,925; 6 dari 55 gambar validasi tertukar
dengan gelas/saset) — perbaikan berikutnya paling menguntungkan dilakukan di sini, bukan
dengan mengganti backbone lagi.

> **Catatan jumlah data.** Salinan lokal berisi **2.753** gambar, sedangkan halaman resmi
> menyebut ±2.900. Penyebabnya sudah diperiksa lewat `train_sources.csv`: CSV memuat
> **2.964 baris** (semua nama unik), 0 gambar di disk yang tidak tercatat, dan **211 baris
> tercatat tetapi tidak ada di disk** (botol −18 · gelas −51 · kertas −30 · saset −85 ·
> styrofoam −27). Jadi CSV adalah catatan saat pengumpulan dan folder `train/` adalah hasil
> **setelah pembersihan** (deduplikasi + penyaringan + pemeriksaan manual asisten) — sesuai
> penjelasan halaman Studio 2. **2.753 adalah jumlah yang benar dan terkini**, dataset tidak
> perlu diunduh ulang, dan pelatihan memakai seluruh 2.753 gambar.

## Pembagian Data (Train / Validasi)

Pembagian dihitung saat melatih, **berstrata** per kelas dengan **seed tetap 42**, jadi
identik setiap kali dijalankan ulang di Colab maupun di laptop.

| | |
|---|---|
| **Latih** | 2.340 gambar (85%) |
| **Validasi (`my_val/`)** | 413 gambar (15%) — botol 78 · gelas 78 · kertas 55 · saset 101 · styrofoam 101 |

`my_val/` hanya pembanding internal antar-run kita sendiri. Karena daftar gambar ujiannya
berbeda dari validation set tersembunyi asisten, angkanya **tidak boleh** diklaim sebagai
perkiraan skor asisten.

## Ke Mana Bobot (`.pth`) Disimpan?

| Mode | Perintah | Lokasi bobot |
|---|---|---|
| **Percobaan** (bawaan) | `python train.py --data "<...>/train" --tag coba_v2s` | `runs/<tag>/<tag>.pth` |
| **Putaran Final** | tambah `--to-submission` | `submission/model.pth` |

Bawaannya menulis ke `runs/`, sehingga tiap percobaan punya berkas sendiri dan `submission/`
tidak pernah rusak karena percobaan. Promosi ke `submission/` dilakukan `experiments/promote.py`,
yang mencadangkan versi lama, menyalin `model.py` + `model.pth` berpasangan, lalu **menguji ulang**
`load_state_dict(strict=True)` dan mengembalikan versi lama bila gagal.

## Catatan Teknis Penting

- **Urutan kelas wajib alfabetis** (sama dengan `ImageFolder` dan `evaluate.py`):
  `0 botol_plastik · 1 gelas_plastik · 2 kertas_kardus · 3 saset_kemasan · 4 styrofoam`
- **`model.py` hanya membangun arsitektur**: `weights=None`, tanpa mengunduh bobot dan tanpa
  memuat `model.pth` (yang memuat adalah `evaluate.py`).
- **Batas teknis:** `model.pth` kelompok ini 81,7 MB (≤ 500 MB) dan evaluasi 413 gambar
  selesai 6,9 detik (≤ 15 menit), jadi tidak perlu GitHub Release / Git LFS.
- **Kompatibilitas `strict=True`:** `model.pth` dihasilkan oleh `get_model()` dari `model.py`
  yang sama, jadi seluruh nama kunci `state_dict` cocok.

## Sumber Daya

- Halaman resmi Studio 2: https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/
- Starter kit resmi: `evaluate.py`, `submission/model.py`, `train_baseline.py`, `README_TEMPLATE.md`
- Materi Pertemuan 4–5 (CNN from scratch, variasi arsitektur) dan Pertemuan 6 (fine-tuning ImageNet)
- Dataset: train set resmi mata kuliah (citra dari internet, keperluan pendidikan non-komersial;
  sumber per gambar di `train_sources.csv`, tidak di-commit)
