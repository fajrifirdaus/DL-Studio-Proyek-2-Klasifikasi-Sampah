# experiments/ — ruang kerja kandidat model

Folder ini untuk **mencoba arsitektur/backbone lain** (mis. ResNet-34, ResNet-50)
tanpa mengacaukan `submission/` yang sudah aman.

| Folder | Isi | Boleh kotor? |
|---|---|---|
| `experiments/` | kandidat arsitektur yang **sedang dicoba** | ✅ boleh |
| `runs/` | hasil tiap percobaan (bobot + `metrics.json`) — tidak di-commit | ✅ boleh |
| `submission/` | **hanya satu** model terbaik, siap dinilai asisten | ❌ wajib bersih |

## Cara mencoba backbone lain (di notebook)

Semua diatur dari **sel KONFIGURASI** (`CFG`), lalu jalankan ulang sel data loader & LATIH:

```python
CFG["arsitektur"] = "experiments/resnet34_model.py"   # bentuk model
CFG["backbone"]   = "resnet34"                        # bobot ImageNet mana
CFG["tag"]        = "coba_resnet34"                   # supaya runs/ tidak menimpa
```

> **Jangan lupa `tag`.** Tanpa tag berbeda, bukti percobaan saling menimpa dan kamu
> kehilangan angka pembandingnya.

**Aman dari kecelakaan:** bawaannya `CFG["tulis_submission"] = False`, jadi bobot
percobaan disimpan ke `runs/<tag>.pth` — `submission/` **tidak tersentuh**.

## Cara mencoba backbone lain (di laptop, CLI)

```bash
python train.py --data "<...>/train" --arsitektur experiments/resnet34_model.py \
                --backbone resnet34 --tag coba_resnet34
```

## Isi berkas kandidat

Cukup **dua fungsi** (lihat `resnet34_model.py` sebagai contoh lengkap):

```python
BACKBONE = "resnet34"          # wajib, kalau ingin bobot ImageNet disalin

def get_model():                # hanya MEMBANGUN arsitektur (weights=None)
    ...
    return model

def get_transform():            # transform validasi; resolusi bebas (224/288/...)
    ...
```

| Hal | Aturan |
|---|---|
| `BACKBONE` | nama model torchvision (`resnet18`, `resnet34`, `resnet50`, …) |
| `get_model()` | `weights=None` — jangan unduh bobot di sini |
| `get_transform()` | resolusi **bebas**; augmentasi latih otomatis menyesuaikan |
| Ganti resolusi | `get_transform()` **dan** latih harus sama — notebook mengurus ini sendiri |

## Kalau kandidatnya MENANG → salin ke `submission/model.py`

Ini **manual, sengaja** — supaya tidak ada yang otomatis menimpa submission:

| # | Langkah |
|---|---|
| 1 | Buka `submission/model.py` |
| 2 | Ganti isi `get_model()` dengan versi kandidat |
| 3 | Pastikan `get_transform()` di `submission/model.py` **sama** dengan kandidat |
| 4 | Set `CFG["tulis_submission"] = True` di notebook, lalu jalankan ulang sel LATIH |
| 5 | Jalankan evaluasi resmi: `python evaluate.py --submission ./submission --data ./my_val` |

> **`model.py` dan `model.pth` WAJIB berpasangan.** Kalau `model.py` jadi ResNet-34
> tetapi `model.pth` masih bobot ResNet-18, evaluasi asisten akan **gagal**
> (`load_state_dict(strict=True)`), dan skor putaran itu hilang.

## Kandidat yang Tersedia

| Berkas | Backbone | Resolusi | Bobot ImageNet | Lolos 100 MB? | Catatan |
|---|---|---|---|---|---|
| `experiments/resnet34_model.py` | resnet34 | 224 | 83,3 MB | ya | transform **sama** dengan ResNet-18 — paling aman dicoba |
| `experiments/efficientnet_v2s_model.py` | efficientnet_v2_s | **384** | 82,7 MB | ya | akurasi asli V2; latih ~3,5x lebih lama, pakai `batch_size` 8–16 |

### Kandidat lain yang tersedia di torchvision (belum dibuatkan berkasnya)

Ukuran di bawah adalah **bobot ImageNet nyata** (diukur dari server PyTorch).
`model.pth` hasil latihan kira-kira sebesar itu.

| Backbone | Bobot | Lolos 100 MB? |
|---|---|---|
| mobilenet_v3_small | 9,8 MB | ya |
| mobilenet_v2 | 13,6 MB | ya |
| regnet_y_400mf | 16,8 MB | ya |
| efficientnet_b0 | 20,5 MB | ya |
| densenet121 | 30,8 MB | ya |
| resnext50_32x4d | 95,8 MB | ya (mepepet) |
| resnet50 | 97,8 MB | ya (mepepet) |
| swin_t / convnext_tiny | 108 MB | **tidak** — perlu GitHub Release |
| resnet101 / vit_b_16 | 170–330 MB | **tidak** — perlu GitHub Release |
| vgg16 | 528 MB | **tidak** — melanggar batas 500 MB aturan |

> **Semua model di atas ada di `torchvision`** — jadi syarat "tanpa timm" tetap terpenuhi.

### Penting: resolusi setiap model BEDA

Kalau resolusi dipaksa seragam, akurasi turun **tanpa pesan error**. Setelan yang
disarankan (dari sumber torchvision):

| Keluarga | Crop | Resize | Interpolasi |
|---|---|---|---|
| ResNet, RegNet, MobileNetV3, DenseNet | 224 | 232 | Bilinear |
| EfficientNet V1-B0 | 224 | 256 | Bilinear |
| **EfficientNet V2-S** | **384** | **384** | Bilinear |
| ConvNeXt | 224 | 230 | Bilinear |
| Swin | 224 | 232 | **Bicubic** |

`train.py` dan notebook **sudah otomatis** menyesuaikan augmentasi latih ke
resolusi kandidat, jadi cukup pastikan `get_transform()` kandidat benar.

## Alur promosi yang aman (otomatis)

`promote.py` memindahkan **KEDUA** berkas sekaligus (`model.py` **dan** `model.pth`)
dalam satu langkah — jadi tidak perlu salin manual satu per satu.

```bash
# 1) latih kandidat -> bobotnya OTOMATIS ke runs/<tag>.pth
python train.py --data "<...>/train" --tag coba_resnet34 \
                --arsitektur experiments/resnet34_model.py --backbone resnet34

# 2) uji saja, tidak mengubah apa pun
python experiments/promote.py --candidate runs/coba_resnet34.pth \
       --arsitektur experiments/resnet34_model.py --dry-run

# 3) kalau menang, promosikan (model.py + model.pth berpasangan)
python experiments/promote.py --candidate runs/coba_resnet34.pth \
       --arsitektur experiments/resnet34_model.py --yes
```

**Yang dilakukan `promote.py` saat promosi:**

| # | Langkah |
|---|---|
| 1 | Ukur skor lama & baru dengan `evaluate.py` resmi (angka dari JSON, bukan tebakan) |
| 2 | Kandidat lebih rendah → promosi **dibatalkan** (paksa: `--force`) |
| 3 | `model.py` + `model.pth` lama dicadangkan ke `runs/_arsip/<tanggal>/` |
| 4 | Salin `model.py` (kalau `--arsitektur` diberikan) + `model.pth` kandidat |
| 5 | **Periksa ulang**: submission baru harus bisa dimuat (`load_state_dict` strict). Kalau gagal → otomatis **dikembalikan** ke versi lama |
| 6 | Catat riwayat ke `runs/_arsip/riwayat.jsonl` |

> Tanpa `--arsitektur`, `promote.py` menganggap arsitekturnya sama dengan
> `submission/model.py` yang sekarang (cukup ganti bobotnya).
>
> Tanpa `--dry-run` atau `--yes`, skrip ini **tidak melakukan apa-apa** — jadi tidak
> ada promosi yang jalan karena kecelakaan.

## Catatan jujur

Ganti backbone **bukan jaminan skor naik**. Baseline resmi sudah 0,92 dan ruang
perbaikannya lebih besar di **cara melatih** (bobot kelas, early stopping, seleksi
macro-F1) — itu yang sudah dikerjakan `train.py`. Backbone lebih besar juga berarti:
waktu latih 1,5–2×, `model.pth` ±85 MB (masih di bawah 100 MB), dan risiko OOM
(lihat `batch_size`).
