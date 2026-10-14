# Notebooks

Folder ini berisi notebook Colab untuk **melatih**, terpisah dari kode utama di root repo.

## `train_colab.ipynb`

Notebook pelatihan lengkap di Google Colab. **Ini tempat melatih dengan GPU gratis** — yang
disimpan ke repo hanya kode dan `submission/model.pth`; dataset tetap di Drive-mu.

### Cara pakai

1. Buka di Colab: **File → Upload notebook**, pilih berkas ini.
2. **Runtime → Change runtime type → T4 GPU** (wajib; tanpa GPU latihan sangat lambat).
3. Siapkan ZIP dataset di Google Drive, mis. `MyDrive/Tugas Studio 2/Dataset/`.
4. **Runtime → Run all**.

### Isi ZIP dataset — hanya butuh folder `train/`

```
NAMA_ZIP.zip
└── train/
    ├── botol_plastik/
    ├── gelas_plastik/
    ├── kertas_kardus/
    ├── saset_kemasan/
    └── styrofoam/
```

**Kode tidak perlu ikut ZIP.** Notebook mencari folder repo di dalam ZIP; kalau tidak ada, ia
otomatis `git clone` dari GitHub (repo publik, kode saja — ±232 KB tanpa bobot). Tanpa
`evaluate.py`, langkah uji resmi tidak bisa dijalankan dan notebook akan memberi tahu.

> **Fallback bawaan — perhatikan.** Kalau `submission/model.py` tidak ditemukan (ZIP tanpa repo
> **dan** clone gagal), notebook menulis salinan bawaan **ResNet-18 + transform 224** di dalam
> Colab supaya tetap bisa jalan. Salinan itu bukan model yang dipakai sekarang — kalau kamu
> melihat cetakan `backbone = resnet18` di sel 5, hentikan dan perbaiki dulu (lihat bagian
> Konfigurasi di bawah).

### Langkah demi langkah (opsi B: panduan juga ada di dalam notebook)

Notebook ini sudah memuat **sel "5b. Cara pakai sel ini — ganti backbone, tag, dan kunci
lain"** tepat setelah sel KONFIGURASI, jadi kamu tidak perlu bolak-balik ke README. Isinya:
resep siap tempel untuk V2-S / ResNet-34 / baseline ResNet-18, tiga aturan yang tidak boleh
dilanggar, tabel kunci `CFG` yang sering dipakai, dan cara memastikan bukan fallback bawaan
yang terpakai.

### Konfigurasi yang harus dipakai sekarang (sel 5)

Model terpasang di `submission/` adalah **EfficientNet V2-S @384**, jadi konfigurasi percobaan
yang benar:

| Kunci `CFG` | Nilai | Kenapa |
|---|---|---|
| `arsitektur` | `"experiments/efficientnet_v2s_model.py"` | bentuk model V2-S + transform 384 |
| `backbone` | `"efficientnet_v2_s"` | bobot ImageNet yang disalin |
| `batch_size` | `32` (diturunkan **otomatis** jadi 8 untuk 384) | 384 tidak muat di batch 32 |
| `tulis_submission` | `False` untuk percobaan; `True` hanya saat putaran Final | jangan rusak `submission/` |
| `tag` | `None` → otomatis `<backbone>_split_seed42` | format ini sama dengan arsip di `runs/` |
| `val_frac` / `kfolds` / `seed` | `0.15` / `0` / `42` | split standar kelompok (413 gambar) |

Salin-tempel untuk percobaan V2-S, lalu **jalankan ulang sel 13 & 14**:

```python
CFG["arsitektur"] = "experiments/efficientnet_v2s_model.py"
CFG["backbone"]   = "efficientnet_v2_s"
CFG["batch_size"] = 8               # 384 px
CFG["tag"]        = "coba_v2s"      # jangan None, supaya tidak menimpa percobaan lain
```

Percobaan lain: `arsitektur = "experiments/resnet34_model.py"` + `backbone = "resnet34"` +
`tag = "coba_resnet34"` (resolusi 224, batch 32).

> **Periksa cetakan `KONFIGURASI` di akhir sel 5.** Baris `arsitektur` dan `backbone` harus
> menunjuk percobaan yang sama — kalau tidak, bobot ImageNet tidak akan cocok dan model dilatih
> hampir dari nol tanpa pesan error.
>
> **Sel "Catatan Penutup" masih mencontohkan `freeze_epochs = 3`** (linear probing), sedangkan
> `CFG` sekarang `0` (full fine-tuning sejak epoch 1 — resep yang menghasilkan angka di bawah).
> Yang berlaku adalah `CFG`. Kalau mau linear probing, ubah `CFG["freeze_epochs"]` lalu jalankan
> ulang sel 13–14.

### Hasil yang sudah terukur (split validasi 413 gambar, seed 42)

| Tag di `runs/` | Kandidat | Resolusi | Epoch terbaik | Macro-F1 | Status |
|---|---|---|---|---|---|
| `resnet18_kf1_seed42` | ResNet-18 **scratch**, K-fold 1 | 224 | — | — | ⚠️ run tidak lengkap (tanpa `metrics.json`) |
| `resnet34_split_seed42` | ResNet-34 pretrained | 224 | 22/30 | 0,9408 | kalah |
| `efficientnet_v2_s_split_seed42` | **EfficientNet V2-S pretrained** | **384** | 16/24 | **0,9618** | ✅ terpasang di `submission/` |

Nama tag otomatis (`<backbone>_split_seed42`) sengaja dibuat sama dengan isi `runs/`, supaya
asal-usul setiap bobot bisa dilacak hanya dari namanya.

> **Catatan jujur.** Angka di atas adalah **epoch terbaik pada himpunan yang dipakai memilih
> epoch**, jadi bias optimistis. Pada 328 gambar validasi yang tidak pernah dipakai seleksi,
> selisih V2-S vs ResNet-34 tinggal +2,7 pp (96,04% vs 93,29%) — masih di bawah ambang noise
> ±4,8 pp untuk val 413 gambar. Jangan menyebut angka ini sebagai perkiraan skor validation set
> tersembunyi; pembanding yang sah adalah kolom skor asisten.

### Pembagian Data (train / validasi)

Tidak ada berkas split yang perlu disiapkan. Pembagiannya **dihitung di notebook** saat melatih,
memakai `seed = 42` yang tetap dan **berstrata** per kelas:

| | |
|---|---|
| Latih | 2.340 gambar (85%) |
| Validasi | 413 gambar (15%) — botol 78 · gelas 78 · kertas 55 · saset 101 · styrofoam 101 |

Hasilnya sama persis setiap kali dijalankan ulang. Ini sudah diuji: split 413 gambar dari
logika `split_single()` di `train.py` dibangun ulang di laptop dan **menghasilkan confusion matrix
serta macro-F1 yang identik** dengan `runs/efficientnet_v2_s_split_seed42/metrics.json` — bukti
bahwa split notebook, `train.py`, dan `my_val/` laptop memang satu pembagian yang sama.

> Angka validasi ini **tidak bisa dibandingkan langsung** dengan angka tim lain (daftar gambar
> ujiannya berbeda). Pembanding yang sah adalah hasil di Google Sheets asisten.
>
> **`--kfolds` / sel 19 bukan alat ukur jujur** di pipeline ini: run K-fold selalu memakai seluruh
> data di luar `my_val/` sebagai data latih, sehingga lipatan validasinya sebagian besar sudah
> pernah dilihat model (terukur 468 dari 553 gambar). Pakai hanya untuk uji sensitivitas.

### Uji wajib setelah melatih (sel 15)

Sel 15 menjalankan `evaluate.py` **resmi milik asisten** pada `my_val/`, jadi hasilnya harus
**sama** dengan macro-F1 di sel 14. Kalau sel ini gagal, submission belum memenuhi kontrak dan
asisten juga akan gagal menjalankannya. Setelah lolos, bobotnya diunduh (sel 18) lalu
dipromosikan lewat `experiments/promote.py` — alurnya ada di `experiments/README.md`.

### Kalau ada masalah

| Gejala | Sebab & tindakan |
|---|---|
| `CUDA out of memory` | Turunkan `CFG["batch_size"]`. Untuk 384 px nilainya sudah diturunkan otomatis ke 8; kalau masih OOM pakai `CFG["limit"] = 1000` untuk uji cepat |
| Folder dataset tidak terdeteksi | Pastikan ZIP memuat `train/` dengan kelima subfolder kelas bernama persis seperti di atas |
| `backbone = resnet18` di cetakan | Notebook memakai fallback bawaan (repo tidak ketemu di ZIP dan clone gagal) — perbaiki dulu, jangan latih |
| Latih sangat lambat | Runtime belum T4 GPU — **Runtime → Change runtime type** |

`torch`, `torchvision`, `pillow`, dan `numpy` sudah tersedia di Colab, jadi tidak ada pemasangan
paket. `antialias=True` yang dipakai `experiments/efficientnet_v2s_model.py` juga aman di Colab
(torchvision Colab jauh di atas 0,15) — keluhan versi ini hanya muncul di venv lokal yang tua.

### Sebelum commit ke repo

| # | Periksa |
|---|---|
| 1 | `evaluate.py` (sel 15) jalan tanpa error |
| 2 | Macro-F1 sel 15 sama dengan sel 14 |
| 3 | `submission/model.pth` ±**78 MB** — di bawah batas 100 MB GitHub, jadi tidak perlu Git LFS |
| 4 | waktu evaluasi di sel 15 jauh di bawah 15 menit (terukur ±12 detik untuk 413 gambar di GPU lokal) |
| 5 | commit `submission/model.py` **dan** `submission/model.pth` bersamaan |
| 6 | jangan commit dataset, `my_val/`, bobot di `runs/` (`.pth`), atau `train_sources.csv` — tetapi berkas bukti `runs/<tag>/{config,history,metrics}.json` + `kurva_pelatihan.png` IKUT di-commit |
| 7 | hasil putaran Final dilatih ulang di 100% data (`--full-data`, tanpa validasi) lalu dipromosikan ulang |
