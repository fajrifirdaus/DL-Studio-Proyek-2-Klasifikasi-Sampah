# Notebooks

Folder ini berisi notebook Colab untuk **melatih**, terpisah dari kode utama di root repo.

## `train_colab.ipynb`

Notebook pelatihan lengkap di Google Colab. **Ini satu-satunya tempat melatih** —
Colab memberi GPU gratis, jadi semua percobaan dijalankan di sini. Yang disimpan
ke repo hanya kode, dan yang dibawa pulang hanya `model.pth`.

### Cara pakai

1. Buka di Colab: **File → Upload notebook**, pilih berkas ini.
2. **Runtime → Change runtime type → T4 GPU** (wajib; kalau tidak ada GPU, latihan sangat lambat).
3. Siapkan ZIP dataset di Google Drive-mu, mis. `MyDrive/Tugas Studio 2/Dataset/`.
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

**Kode tidak perlu ikut ZIP.** Kalau folder repo tidak ditemukan, notebook otomatis
melakukan `git clone` dari GitHub (repo publik, kode saja, ±133 KB). Jadi cukup dataset
di Drive dan notebook ini.

> Tanpa `submission/model.py`, notebook membuat **salinan bawaan** (ResNet-18 + transform
> 224) langsung di dalam Colab. Tanpa `evaluate.py`, langkah uji resmi tidak bisa
> dijalankan (notebook akan memberi tahu).

### Pembagian Data (train / validasi)

Tidak ada berkas split yang perlu disiapkan. Pembagiannya **dihitung di notebook** saat
melatih, memakai `seed = 42` yang tetap dan **berstrata** per kelas — hasilnya sama
persis setiap kali dijalankan ulang:

| | |
|---|---|
| Latih | 2.340 gambar (85%) |
| Validasi | 413 gambar (15%) — botol 78 · gelas 78 · kertas 55 · saset 101 · styrofoam 101 |

> Angka ini **tidak bisa dibandingkan langsung** dengan angka tim lain (daftar gambar
> ujiannya berbeda). Pembanding yang sah adalah hasil di Google Sheets asisten.
