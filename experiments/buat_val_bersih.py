"""Bangun my_val_bersih/: subset my_val/ TANPA gambar yang hampir kembar dengan data latih.

KENAPA?
-------
Preview 1: skor lokal 0,9618 tetapi skor asisten 0,9361. Sebagian my_val/ hampir
kembar dengan gambar latih (foto benda/sumber yang sama), jadi my_val/ lebih mudah
daripada val tersembunyi asisten yang berisi gambar baru. Subset ini membuang
gambar val yang punya "kembaran" di data latih, supaya pemilihan model lebih jujur.

CARA KERJA
----------
dHash 64-bit (perbedaan kecerahan piksel bertetangga pada gambar 9x8 abu-abu).
Gambar val dibuang bila jarak Hamming ke gambar latih TERDEKAT <= --ambang (bawaan 8).
Data latih = train/ dikurangi my_val/ (sama dengan split train.py, seed 42).
Folder hasil berisi SALINAN gambar (bukan symlink) agar evaluate.py bisa langsung
membacanya:   python evaluate.py --submission ./submission --data ./my_val_bersih

Pemakaian (dari root repo):
    python experiments/buat_val_bersih.py --train ./train --val ./my_val
my_val_bersih/ tidak di-commit (berisi gambar dataset; lihat .gitignore).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(p: Path) -> np.ndarray:
    im = Image.open(p).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def kumpul(d: Path) -> dict[str, Path]:
    return {f"{c.name}/{f.name}": f for c in sorted(d.iterdir()) if c.is_dir()
            for f in sorted(c.iterdir()) if f.suffix.lower() in EXT}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="./train")
    ap.add_argument("--val", default="./my_val")
    ap.add_argument("--out", default="./my_val_bersih")
    ap.add_argument("--ambang", type=int, default=8)
    a = ap.parse_args()

    val, semua = kumpul(Path(a.val)), kumpul(Path(a.train))
    latih = [f for k, f in semua.items() if k not in val]
    if len(semua) - len(latih) != len(val):
        raise SystemExit("[GAGAL] my_val/ bukan subset train/ - jalankan train.py dulu agar split sama")
    hl = np.stack([dhash(f) for f in latih])

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    simpan, buang = [], []
    for k, f in val.items():
        d = int((hl != dhash(f)).sum(1).min())
        (buang if d <= a.ambang else simpan).append((k, d))
    for k, _ in simpan:
        tujuan = out / k
        tujuan.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(val[k], tujuan)

    per_kelas = {}
    for k, _ in simpan:
        per_kelas[k.split("/")[0]] = per_kelas.get(k.split("/")[0], 0) + 1
    info = {"ambang_hamming": a.ambang, "val_asal": len(val), "dibuang": len(buang),
            "tersisa": len(simpan), "per_kelas": per_kelas}
    (out / ".bersih.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
