# Studio 2 — Kompetisi Klasifikasi Sampah

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

**Nama kelompok:** `<15/DUO FF>`

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Fadhilah Akbar | 123140003 | `train.py` (perbaikan celah baseline: seleksi macro-F1, bobot kelas, early stopping), `notebooks/train_colab.ipynb` (alur Colab), `experiments/` (kandidat ResNet-34 & EfficientNet V2-S, `promote.py`), pelatihan kedua kandidat | 
| 2  | Muhammad Fajri Firdaus  | 123140050 | `submission/` (EVA-02 Small @336 + bobot terlatih), `evaluate.py` (uji kontrak submission), `README.md` + dokumentasi subfolder, verifikasi bebas kontaminasi split validasi (diagnostik kebocoran & McNemar) |

Repo: https://github.com/fajrifirdaus/DL-Studio-Proyek-2-Klasifikasi-Sampah

## Ringkasan Pendekatan

| Butir | Isi |
|---|---|
| **Arsitektur / backbone** | **EVA-02 Small @336** (`timm` `eva02_small_patch14_336`), pretrained MIM ImageNet-22k → fine-tune ImageNet-1k (`mim_in22k_ft_in1k`); sejak Preview 2. Preview 1: EfficientNet V2-S (torchvision) |
| **Strategi pelatihan** | Full fine-tuning (seluruh layer dilatih); AdamW lr 2e-5, weight decay 0,05, warmup 1 epoch + cosine decay, gradient clipping 1,0. Resep dipilih dari run split (puncak epoch 13, epoch 14–15 sama 0,9778); model Final/terpasang dilatih ulang dengan resep itu pada seluruh data (`--full-data`, 15 epoch, bobot = epoch terakhir). Skor lokal di tabel Hasil berasal dari run split karena model terpasang tidak punya data uji lokal |
| **Augmentasi** | RandomResizedCrop(336, 0.60–1.00), horizontal flip, ColorJitter(0.3/0.3/0.3/0.05), RandomErasing p=0.25 + normalisasi CLIP (sama dengan evaluasi) |
| **Penanganan kelas tidak seimbang** | Bobot kelas `N / (C × n_c)` pada CrossEntropyLoss, dihitung hanya dari data latih |
| **Label noise** | Label smoothing 0,1 |
| **Data tambahan** | **Tidak ada** — hanya train set dari starter kit; dataset tidak diunggah ulang |
| **Ukuran input & preprocessing** | `Resize(336, bicubic)` → `CenterCrop(336)` → `ToTensor` → normalisasi CLIP (`mean [0.4815,0.4578,0.4082]`, `std [0.2686,0.2613,0.2758]`, sesuai pretrain EVA-02) |
| **Paket** | `torch`, `torchvision`, `timm>=0.9`, `pillow`, `numpy` (semua paket dasar environment asisten) |
| **Ukuran `submission/model.pth`** | 43,5 MB — `state_dict` disimpan fp16 (prediksi identik dengan fp32: 0 dari 413 berbeda), dimuat ke model fp32 dengan `strict=True`; di bawah batas GitHub 100 MB, jadi tanpa Release/LFS |
| **Waktu evaluasi** | 413 gambar = **8,5 detik** GPU / **36,4 detik CPU** lokal (setara ±65 detik CPU untuk ±740 gambar) — jauh di bawah batas aturan 15 menit |

## Hasil

Kolom "lokal" = macro-F1 epoch terbaik pada split validasi lokal sendiri (413 gambar, seed 42). Kolom "asisten" = skor yang diumumkan pada validation set tersembunyi.

| Putaran | Commit | Backbone | Macro-F1 lokal | Akurasi lokal | Macro-F1 asisten |
|---|---|---|---|---|---|
| Preview 1 (14 Okt) | [commit `submission/`](https://github.com/fajrifirdaus/DL-Studio-Proyek-2-Klasifikasi-Sampah/commits/main/submission/model.py) | EfficientNet V2-S | 0,9618 | 0,9661 | **0,9361** (akurasi 0,9379; peringkat 7 dari 10 yang dinilai; baseline 0,9159, juara 0,9590) |
| Preview 2 (21 Okt) | [commit `submission/`](https://github.com/fajrifirdaus/DL-Studio-Proyek-2-Klasifikasi-Sampah/commits/main/submission/model.py) | EVA-02 Small @336 | 0,9781 | 0,9806 | *(menunggu pengumuman)* |
| Final (25 Okt) | *(belum)* | — | — | — | — |

**Skor lokal vs skor asisten (Preview 1).** Skor asisten 0,9361 berada 2,6 poin di bawah skor
lokal 0,9618 pada model yang sama (commit `3a5814f`, terakhir sebelum tenggat 14 Okt 23:59).
Selisih ini nyata (akurasi 96,61% vs 93,79%, z = 2,24), bukan sekadar noise. Yang sudah diperiksa:
(1) skor lokal adalah epoch terbaik yang dipilih pada `my_val/` itu sendiri, jadi bias optimistis;
(2) dugaan "my_val terlalu mirip data latih" **tidak terbukti**: setelah 73 gambar val yang hampir
kembar dengan data latih dibuang (dHash, jarak Hamming ≤ 8; `experiments/buat_val_bersih.py`),
skor semua kandidat justru **naik** (mis. V2-S 0,9618 → 0,9761 pada 340 gambar sisa). Jadi gambar
yang dibuang itu lebih sulit, bukan lebih mudah, dan penurunan skor asisten tidak berasal dari
kebocoran split. Penjelasan yang tersisa: val asisten berasal dari distribusi gambar yang berbeda
(sumber/foto lain), yang tidak bisa ditiru dengan potongan train set kita sendiri.

Tautan pada kolom **Commit** menunjuk riwayat commit `submission/model.py` — berisi hash commit
Preview 1 beserta seluruh perubahan sesudahnya. Hash pastinya bisa dilihat dengan
`git log -1 --format=%H -- submission/model.py`.

Sumber angka: `runs/<tag>/metrics.json` (berkas JSON-nya di-commit sebagai bukti) dan keluaran resmi
`python evaluate.py --submission ./submission --data ./my_val` (diulang persis seperti asisten).

### Perbandingan backbone yang diuji

Semua kandidat dilatih pada **split validasi yang sama** (413 gambar, seed 42), seleksi berdasarkan macro-F1.
Tiap run (berhasil maupun gagal) punya commit sendiri berisi pengaturan, perubahan dan alasannya, serta hasil.

**Preview 1**

| Kandidat | Resolusi | Epoch terbaik | Macro-F1 | Akurasi | F1 kertas_kardus |
|---|---|---|---|---|---|
| ResNet-34 | 224 | 22/30 | 0,9408 | 94,19% | 0,925 |
| **EfficientNet V2-S** | **384** | **16/24** | **0,9618** | **96,61%** | 0,925 |

**Preview 2** (backbone `timm`; resep ViT/ConvNeXt: AdamW wd 0,05, warmup 1, clip-grad 1,0)

| Kandidat | Resolusi | lr / batch | Epoch terbaik | Macro-F1 | Akurasi | F1 kertas_kardus | Status |
|---|---|---|---|---|---|---|---|
| **EVA-02 Small** | **336** | 2e-5 / 8 | 13/15 | **0,9781** | **98,06%** | 0,953 | **dipilih** (arsitektur + resep) |
| EVA-02 Small (seed latih 7) | 336 | 2e-5 / 8 | 5/10 | 0,9814 | 98,31% | 0,964 | puncak sesaat (ep4 0,9549 → ep5 0,9814 → ep6 0,9696); rata-rata ep ≥ 6 di bawah seed 42 |
| CAFormer-S18 | 384 | 5e-5 / 8 | 7/12 | 0,9755 | 97,82% | 0,945 | tidak lebih baik |
| ConvNeXt V2-Base | 224 | 5e-5 / 8 | 10/14 | 0,9739 | 97,58% | 0,953 | bobot 175 MB, tidak dipilih |
| TinyViT-21M | 384 | 5e-5 / 8 | 8/13 | 0,9734 | 97,58% | 0,944 | tidak lebih baik |
| ConvNeXt V2-Tiny | 384 | 5e-5 / 8 | 8/13 | 0,9725 | 97,58% | 0,943 | — |
| Ensemble Tiny + V2-S | 384 | — | — | 0,9672 | 97,09% | 0,935 | gagal (< Tiny saja) |
| DINOv2 ViT-S | 224 | 2e-5 / 16 | 14/15 | 0,9539 | 95,88% | 0,899 | gagal (< V2-S) |
| ConvNeXt V2-Tiny | 384 | 3e-4 / 8 | — | turun 0,94→0,87 | — | — | dihentikan (lr terlalu besar) |
| ConvNeXt V2-Tiny | 384 | 1e-4 / 8 | — | runtuh 0,90→0,11 di ep 4 | — | — | dihentikan |

EVA-02 Small dipasang ke `submission/` untuk Preview 2 (+1,6 poin macro-F1 atas V2-S). Uji
McNemar: vs V2-S 9 vs 3 gambar (p = 0,146); vs ConvNeXt V2-Tiny 6 vs 4 (p = 0,75). Jadi
keunggulannya **belum signifikan** secara statistik pada 413 gambar; EVA-02 dipilih karena skor
tertinggi, paling ringan (21,7 jt parameter, bobot 43,5 MB), dan naik stabil tanpa runtuh.
Pelajaran utama: cara/data pretrain (MIM, ImageNet-22k) lebih berpengaruh daripada ukuran model
(ConvNeXt V2-Base 87,7 jt hanya +0,0014 atas Tiny 27,9 jt).

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

# Latih model terpasang (EVA-02 Small @336); bobot ke runs/<tag>/<tag>.pth, submission/ tidak tersentuh
python train.py --data "<...>/train" --arsitektur experiments/eva02_small_model.py \
                --epochs 15 --lr 2e-5 --weight-decay 0.05 --warmup 1 --clip-grad 1.0 \
                --start-epoch 6 --patience 5 --batch-size 8 --tag eva02_small_336_lr2e-5_seed42

# Latih ulang pada 100% data dengan resep yang sama (tanpa validasi; bobot = epoch ke-15).
# Ini model yang terpasang di submission/.
python train.py --data "<...>/train" --arsitektur experiments/eva02_small_model.py --full-data \
                --epochs 15 --lr 2e-5 --weight-decay 0.05 --warmup 1 --clip-grad 1.0 \
                --batch-size 8 --tag eva02_small_336_lr2e-5_full_seed42

# Simpan bobot dalam fp16 (43,5 MB; prediksi identik dengan fp32), lalu uji dan promosikan
python -c "import torch; t='runs/eva02_small_336_lr2e-5_full_seed42/eva02_small_336_lr2e-5_full_seed42'; s=torch.load(t+'.pth', weights_only=True); torch.save({k: v.half() if v.is_floating_point() else v for k, v in s.items()}, t+'_fp16.pth')"
python experiments/promote.py --candidate runs/eva02_small_336_lr2e-5_full_seed42/eva02_small_336_lr2e-5_full_seed42_fp16.pth \
       --arsitektur experiments/eva02_small_model.py --dry-run
python experiments/promote.py --candidate runs/eva02_small_336_lr2e-5_full_seed42/eva02_small_336_lr2e-5_full_seed42_fp16.pth \
       --arsitektur experiments/eva02_small_model.py --yes

# WAJIB: uji dengan skrip evaluasi resmi milik asisten
python evaluate.py --submission ./submission --data ./my_val
```

Eksperimen pendukung (RPS): `--freeze-epochs 5` (linear probing → full fine-tuning),
`--scratch` (tanpa bobot ImageNet), `--no-class-weight` (ablasi bobot kelas),
`--kfolds 5 --fold 1..5` (lihat Catatan jujur 2). Putaran Final: `--full-data` (lihat di atas)
melatih ulang di 100% data. Opsi lain: `--split-seed` (seed latih beda, `my_val` sama), `--train-frac` (kurva skor vs jumlah data), `--clip-grad`. Setiap run menulis `runs/<tag>/{config,history,metrics}.json`.

### Perubahan resolusi = perubahan `submission/model.py`

| Resolusi latih | `get_transform()` di `model.py` |
|---|---|
| 224 (ResNet, MobileNet, DenseNet, ConvNeXt) | `Resize(256)` + `CenterCrop(224)` |
| 384 (EfficientNet V2-S Preview 1, ConvNeXt V2-Tiny) | `Resize(384)` + `CenterCrop(384)` |
| **336 (dipakai sekarang, EVA-02 Small)** | `Resize(336, bicubic)` + `CenterCrop(336)` + normalisasi CLIP |

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
│   ├── efficientnet_v2s_model.py # EfficientNet V2-S (384) - Preview 1
│   ├── eva02_small_model.py      # EVA-02 Small (336, timm) -> TERPILIH Preview 2
│   ├── convnextv2_tiny_model.py  # ConvNeXt V2-Tiny (384, timm)
│   ├── convnextv2_base_model.py  # ConvNeXt V2-Base (224, timm)
│   ├── dinov2_vits_model.py      # DINOv2 ViT-S/14 (224, timm)
│   ├── caformer_s18_model.py     # CAFormer-S18 (384, timm)
│   ├── tiny_vit_21m_model.py     # TinyViT-21M (384, timm)
│   ├── buat_val_bersih.py        # my_val_bersih/: val tanpa gambar kembar data latih
│   ├── ensemble_cnxv2t_v2s_model.py + buat_ensemble.py  # ensemble Tiny + V2-S
│   ├── promote.py                # uji kandidat -> promosi ke submission/ kalau menang
│   └── README.md
├── submission/
│   ├── model.py                  # get_model() EVA-02 Small + get_transform() 336
│   ├── model.pth                 # state_dict EVA-02 Small (dilatih 100% data), fp16 (43,5 MB)
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
kelas dengan F1 terendah pada **semua** kandidat (V2-S 0,925; EVA-02 0,953 — 4 dari 55 gambar
validasi masih tertukar dengan gelas/saset/styrofoam). Perbaikan berikutnya paling menguntungkan
dilakukan di sini.

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
- **`model.py` hanya membangun arsitektur**: `timm.create_model(..., pretrained=False)`
  (padanan `weights=None` di torchvision), tanpa mengunduh bobot dan tanpa memuat `model.pth`
  (yang memuat adalah `evaluate.py`). Diuji dengan jaringan diblokir: 0 percobaan koneksi.
  Bobot ImageNet hanya diunduh saat **latih** lewat `muat_bobot_pretrained()` yang dipanggil `train.py`.
- **Batas teknis:** `model.pth` kelompok ini 43,5 MB (≤ 500 MB) dan evaluasi 413 gambar
  selesai 36,4 detik di CPU (≤ 15 menit), jadi tidak perlu GitHub Release / Git LFS.
- **Kompatibilitas `strict=True`:** `model.pth` dihasilkan oleh `get_model()` dari `model.py`
  yang sama, jadi seluruh nama kunci `state_dict` cocok.

## Sumber Daya

- Halaman resmi Studio 2: https://informatika-itera.github.io/deep-learning-IF25-40401/studio-2/
- Starter kit resmi: `evaluate.py`, `submission/model.py`, `train_baseline.py`, `README_TEMPLATE.md`
- Materi Pertemuan 4–5 (CNN from scratch, variasi arsitektur) dan Pertemuan 6 (fine-tuning ImageNet)
- Dataset: train set resmi mata kuliah (citra dari internet, keperluan pendidikan non-komersial;
  sumber per gambar di `train_sources.csv`, tidak di-commit)
