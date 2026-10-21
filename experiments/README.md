# experiments/ — ruang kerja kandidat model

Folder ini untuk **mencoba arsitektur/backbone lain** tanpa mengacaukan `submission/`.

| Folder | Isi | Boleh kotor? |
|---|---|---|
| `experiments/` | berkas kandidat + `promote.py` | ✅ boleh |
| `runs/<tag>/` | hasil tiap percobaan: `config.json`, `history.json`, `metrics.json`, `kurva_pelatihan.png`, `<tag>.pth` — tidak di-commit | ✅ boleh |
| `runs/_arsip/` | cadangan submission lama + `riwayat.jsonl` — hanya `riwayat.jsonl` yang di-commit, arsip bobotnya tidak | ✅ boleh |
| `submission/` | **hanya satu** model terbaik, siap dinilai asisten | ❌ wajib bersih |

## Status Percobaan Saat Ini

Angka di bawah diukur pada **split validasi lokal yang sama** (413 gambar, `seed 42`, `val_frac 0.15`),
seleksi berdasarkan macro-F1. Batch otomatis: 32 untuk 224 px, 8 untuk 384 px.

| Tag di `runs/` | Kandidat | Resolusi | Epoch terbaik | Macro-F1 | Akurasi | Status |
|---|---|---|---|---|---|---|
| `resnet18_kf1_seed42` | ResNet-18 **scratch** (`--scratch`, K-fold 1) | 224 | — | — | — | ⚠️ tidak lengkap — tanpa `metrics.json`, tidak dipakai |
| `resnet34_split_seed42` | ResNet-34 pretrained | 224 | 22/30 | 0,9408 | 94,19% | kalah |
| `efficientnet_v2_s_split_seed42` | EfficientNet V2-S pretrained | 384 | 16/24 | 0,9618 | 96,61% | Preview 1 (skor asisten 0,9361) |
| `eva02_small_336_lr2e-5_seed42` | **EVA-02 Small pretrained (timm)** | **336** | 13/15 | **0,9781** | **98,06%** | ✅ **arsitektur + resep terpasang** (bobot = latih ulang 100% data, `eva02_small_336_lr2e-5_full_seed42`) |
| `eva02_small_336_lr2e-5_seed7` | EVA-02 Small, seed latih 7 | 336 | 5/10 | 0,9814 | 98,31% | puncak sesaat, tidak dipilih |
| `caformer_s18_384_lr5e-5_seed42` | CAFormer-S18 | 384 | 7/12 | 0,9755 | 97,82% | tidak lebih baik |
| `convnextv2_base_224_split_seed42` | ConvNeXt V2-Base | 224 | 10/14 | 0,9739 | 97,58% | bobot 175 MB, tidak dipilih |
| `tiny_vit_21m_384_lr5e-5_seed42` | TinyViT-21M | 384 | 8/13 | 0,9734 | 97,58% | tidak lebih baik |
| `convnextv2_tiny_384_split_seed42` | ConvNeXt V2-Tiny | 384 | 8/13 | 0,9725 | 97,58% | — |
| `dinov2_vits_224_lr2e-5_seed42` | DINOv2 ViT-S | 224 | 14/15 | 0,9539 | 95,88% | gagal |
| `ensemble_cnxv2t_v2s` | Ensemble Tiny + V2-S | 384 | — | 0,9672 | 97,09% | gagal (< Tiny saja) |
| `eva02_small_336_frac{0.25,0.50,0.75}_seed42` | EVA-02 Small, 25/50/75% data latih | 336 | — | 0,9488 / 0,9716 / 0,9685 | 95,16 / 97,34 / 97,09% | kurva data vs skor |

Promosi terakhir: 21 Okt 2026 17:59 → `submission/model.py` + `submission/model.pth` =
EVA-02 Small @336, bobot dari run `eva02_small_336_lr2e-5_full_seed42` (seluruh data, 15 epoch,
fp16 43,5 MB). Skor lokal di tabel memakai run split (`seed42`) dengan arsitektur dan resep yang
sama, karena model terpasang ikut melatih `my_val`. Arsip submission lama di `runs/_arsip/`,
riwayat di `runs/_arsip/riwayat.jsonl`. Uji `evaluate.py` resmi dari klon bersih:
GPU 8,5 detik, CPU 37,0 detik untuk 413 gambar (batas aturan 15 menit).

> **Catatan jujur:** 0,9408 dan 0,9618 adalah **epoch terbaik pada himpunan yang dipakai
> memilih epoch**, jadi bias optimistis. Pada 328 gambar validasi yang benar-benar tidak
> pernah dipakai seleksi, selisihnya +2,7 pp (96,04% vs 93,29%) — masih di bawah ambang
> noise ±4,8 pp untuk val sekecil ini. Karena itu jangan menyebut angka ini sebagai
> perkiraan skor validation set tersembunyi. Rinciannya di `README.md` root.
>
> **`--kfolds 5` bukan alat ukur jujur di pipeline ini.** Run K-fold selalu memakai seluruh
> data di luar `my_val/` sebagai data latih, sehingga lipatan validasinya sebagian besar
> sudah pernah dilihat model (terukur 468 dari 553 gambar). Pakai K-fold hanya untuk uji
> sensitivitas; untuk estimasi jujur, latih ulang dari nol pada data latih lipatan itu.

## Cara mencoba backbone lain (notebook atau CLI)

Semua diatur dari **sel KONFIGURASI** (`CFG`) di notebook, lalu jalankan ulang sel data loader & LATIH:

```python
CFG["arsitektur"] = "experiments/resnet34_model.py"   # bentuk model
CFG["backbone"]   = "resnet34"                        # bobot ImageNet mana
CFG["tag"]        = "coba_resnet34"                   # supaya runs/ tidak menimpa
```

Di laptop (CLI):

```bash
python train.py --data "<...>/train" --arsitektur experiments/resnet34_model.py \
                --backbone resnet34 --tag coba_resnet34

# EfficientNet V2-S: resolusi 384 -> pakai batch 8 (auto kalau tidak disebut)
python train.py --data "<...>/train" --arsitektur experiments/efficientnet_v2s_model.py \
                --backbone efficientnet_v2_s --batch-size 8 --tag coba_v2s
```

Bawaannya `CFG["tulis_submission"] = False` (notebook) / tanpa `--to-submission` (CLI), jadi bobot
percobaan disimpan ke `runs/<tag>/<tag>.pth` — `submission/` **tidak tersentuh**.

> **Jangan lupa `tag`.** Tanpa tag berbeda, bukti percobaan saling menimpa dan kamu kehilangan angka pembandingnya.

## Isi berkas kandidat

Cukup **dua fungsi** (lihat `resnet34_model.py` sebagai contoh lengkap):

```python
BACKBONE = "resnet34"          # wajib, kalau ingin bobot ImageNet disalin

def get_model():                # hanya MEMBANGUN arsitektur (weights=None)
    ...
    return model

def get_transform():            # transform validasi; resolusi bebas (224/384/...)
    ...
```

| Hal | Aturan |
|---|---|
| `BACKBONE` | nama model torchvision (`resnet18`, `resnet34`, `efficientnet_v2_s`, …) |
| `get_model()` | `weights=None` — jangan unduh bobot di sini |
| `get_transform()` | resolusi **bebas**; augmentasi latih otomatis menyesuaikan |
| Ganti resolusi | `get_transform()` **dan** latih harus sama — notebook/`train.py` memeriksa ini |

## Alur promosi yang aman (otomatis) — dipakai untuk memilih submission sekarang

`promote.py` memindahkan **KEDUA** berkas sekaligus (`model.py` **dan** `model.pth`) dalam satu langkah.

**Prasyarat:** folder validasi `my_val/` ada di root repo. Bagian ini pernah bikin perintah gagal —
kalau `my_val/` belum ada (mis. repo baru di-clone; `my_val/` di-`.gitignore`), bangun dulu dengan
menjalankan pelatihan sekali dengan epoch minimal — tidak perlu `--limit`, karena `--limit`
mengubah isi split:

```bash
python train.py --data "<...>/train" --epochs 1        # sekali saja -> my_val/ (413 gambar)
```

Hasil pelatihan sekali itu tidak dipakai (bobotnya masuk `runs/<tag>/`); yang dibutuhkan hanya
folder `my_val/`-nya, dan `submission/` tetap tidak tersentuh.

```bash
# 1) latih kandidat -> bobotnya OTOMATIS ke runs/<tag>/<tag>.pth (submission/ aman)
python train.py --data "<...>/train" --arsitektur experiments/efficientnet_v2s_model.py \
                --backbone efficientnet_v2_s --batch-size 8 --tag coba_v2s

# 2) uji saja, tidak mengubah apa pun
python experiments/promote.py --candidate runs/coba_v2s/coba_v2s.pth \
       --arsitektur experiments/efficientnet_v2s_model.py --dry-run

# 3) kalau menang, promosikan (model.py + model.pth berpasangan)
python experiments/promote.py --candidate runs/coba_v2s/coba_v2s.pth \
       --arsitektur experiments/efficientnet_v2s_model.py --yes
```

> **Bentuk jalur bobot: `runs/<tag>/<tag>.pth`** (dua kali tag). Sesuaikan `--candidate` dengan
> tag yang kamu pakai — salin persis dari baris `bobot :` yang dicetak `train.py` di akhir run.

**Yang dilakukan `promote.py` saat promosi:**

| # | Langkah |
|---|---|
| 1 | Ukur skor lama & baru dengan `evaluate.py` resmi (angka dari JSON `--out`, bukan tebakan dari teks) |
| 2 | Kandidat lebih rendah → promosi **dibatalkan** (paksa: `--force`) |
| 3 | `model.py` + `model.pth` lama dicadangkan ke `runs/_arsip/<cap-waktu>/` |
| 4 | Salin `model.py` (kalau `--arsitektur` diberikan) + `model.pth` kandidat |
| 5 | **Periksa ulang**: submission baru harus bisa dimuat (`load_state_dict` `strict=True`). Kalau gagal → otomatis **dikembalikan** ke versi lama |
| 6 | Catat riwayat ke `runs/_arsip/riwayat.jsonl` |

Tanpa `--arsitektur`, `promote.py` menganggap arsitekturnya sama dengan `submission/model.py` yang
sekarang (cukup ganti bobotnya). Tanpa `--dry-run` atau `--yes`, skrip ini **tidak melakukan apa-apa**.

## Setelah promosi: dua perintah wajib

```bash
# a) ulang dengan skrip evaluasi RESMI (persis seperti asisten)
python evaluate.py --submission ./submission --data ./my_val

# b) commit DUA berkas sekaligus, jangan terpisah
git add submission/model.py submission/model.pth
```

> **Prasyarat lingkungan (penyebab paling sering `ModuleNotFoundError: No module named 'torch'`).**
> `python` harus menunjuk venv yang sudah dipasang `pip install -r requirements.txt`. Periksa dengan
> `python -c "import torch; print(torch.__version__)"`. Kalau gagal, aktifkan venv repo
> (`.\.venv\Scripts\Activate.ps1`) atau panggil langsung python venv-nya, mis.
> `& "C:\path\ke\venv\Scripts\python.exe" evaluate.py --submission ./submission --data ./my_val`.
> Ini bukan masalah kode — `evaluate.py` tidak diubah sama sekali.

## Kandidat yang Tersedia

| Berkas | Backbone | Resolusi | Bobot ImageNet | Lolos 100 MB? | Catatan |
|---|---|---|---|---|---|
| `experiments/resnet34_model.py` | resnet34 | 224 | 83,3 MB | ya | transform **sama** dengan ResNet-18 — paling aman dicoba; terukur macro-F1 0,9408 |
| `experiments/efficientnet_v2s_model.py` | efficientnet_v2_s | **384** | 82,7 MB | ya | ✅ **terpilih**; macro-F1 0,9618, latih ~3,5× lebih lama, `batch_size` 8 |

### Kandidat lain yang tersedia di torchvision (belum dibuatkan berkasnya)

Ukuran di bawah adalah bobot ImageNet nyata (diukur dari server PyTorch); `model.pth` hasil latihan kira-kira sebesar itu.

| Backbone | Bobot | Lolos 100 MB? |
|---|---|---|
| mobilenet_v3_small | 9,8 MB | ya |
| mobilenet_v2 | 13,6 MB | ya |
| regnet_y_400mf | 16,8 MB | ya |
| efficientnet_b0 | 20,5 MB | ya |
| densenet121 | 30,8 MB | ya |
| resnet18 (baseline resmi) | 44,7 MB | ya |
| resnext50_32x4d | 95,8 MB | ya (mepepet) |
| resnet50 | 97,8 MB | ya (mepepet) |
| swin_t / convnext_tiny | 108 MB | **tidak** — perlu GitHub Release |
| resnet101 / vit_b_16 | 170–330 MB | **tidak** — perlu GitHub Release |
| vgg16 | 528 MB | **tidak** — melanggar batas 500 MB aturan |

> **Semua model di tabel ini ada di `torchvision`.** Kandidat Preview 2 memakai `timm` (paket dasar
> environment asisten, dicantumkan di `submission/requirements.txt`); `get_model()`-nya memakai
> `pretrained=False`, jadi tidak mengunduh bobot saat evaluasi.

### Penting: resolusi setiap model BEDA

Kalau resolusi dipaksa seragam, akurasi turun **tanpa pesan error**. Setelan yang disarankan (dari sumber torchvision):

| Keluarga | Crop | Resize | Interpolasi |
|---|---|---|---|
| ResNet, RegNet, MobileNetV3, DenseNet | 224 | 232 | Bilinear |
| EfficientNet V1-B0 | 224 | 256 | Bilinear |
| **EfficientNet V2-S** | **384** | **384** | Bilinear |
| ConvNeXt | 224 | 230 | Bilinear |
| Swin | 224 | 232 | **Bicubic** |

`train.py` dan notebook **sudah otomatis** menyesuaikan augmentasi latih ke resolusi kandidat,
jadi cukup pastikan `get_transform()` kandidat benar. Batch diturunkan otomatis mengikuti resolusi
(224→32, 256→24, 288→16, 320→12, **384→8**, 448→6, 512→4), sehingga 384 aman di T4 16 GB.

## Catatan jujur

Ganti backbone **bukan jaminan skor naik** — dua kandidat di sini menang hanya +2,1 poin dan itu
masih di dalam pita noise val 413 gambar. Baseline resmi 0,92 dan **ruang perbaikan terbesar ada di
cara melatih**, bukan di arsitektur. Sisa pekerjaan yang paling menguntungkan:

1. **`kertas_kardus` adalah kelas terlemah pada KEDUA kandidat** (F1 0,925; 6 dari 55 gambar validasi
   tertukar dengan gelas/saset). Perbaikan di sana menaikkan macro-F1 paling cepat.
2. **Eksperimen pendukung RPS belum dijalankan** dengan `train.py` kelompok ini: `--freeze-epochs 5`
   (linear probing → full fine-tuning), `--scratch` (tanpa bobot ImageNet), `--no-class-weight`
   (ablasi bobot kelas). Belum ada angkanya, jadi jangan diklaim di README sampai diukur.
3. **Putaran Final:** model terpasang sudah dilatih ulang di 100% data (`--full-data`, 15 epoch);
   angka `runs/*split*` tetap dari 85% data.
