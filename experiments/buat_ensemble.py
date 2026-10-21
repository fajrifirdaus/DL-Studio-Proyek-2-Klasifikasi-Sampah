"""Gabungkan bobot dua run terlatih menjadi satu state_dict untuk model ensemble.

Pemakaian (dari root repo):
    python experiments/buat_ensemble.py
    python experiments/buat_ensemble.py --fp16      # simpan setengah presisi

Keluaran: runs/ensemble_cnxv2t_v2s/ensemble_cnxv2t_v2s.pth
Bobot fp16 tetap dimuat ke model fp32 dengan strict=True (dicek di akhir).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SUMBER = {
    "cnx.": ROOT / "runs/convnextv2_tiny_384_split_seed42/convnextv2_tiny_384_split_seed42.pth",
    "v2s.": ROOT / "runs/efficientnet_v2_s_split_seed42/efficientnet_v2_s_split_seed42.pth",
}
ARSITEKTUR = ROOT / "experiments/ensemble_cnxv2t_v2s_model.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs/ensemble_cnxv2t_v2s/ensemble_cnxv2t_v2s.pth"))
    a = ap.parse_args()

    gabung = {}
    for awalan, jalur in SUMBER.items():
        if not jalur.is_file():
            raise SystemExit(f"[GAGAL] bobot tidak ada: {jalur}")
        sd = torch.load(jalur, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            gabung[awalan + k] = v.half() if (a.fp16 and v.is_floating_point()) else v

    spec = importlib.util.spec_from_file_location("ens", ARSITEKTUR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.get_model().load_state_dict(gabung, strict=True)   # gagal di sini = kunci tak cocok

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gabung, out)
    print(f"[OK] {len(gabung)} tensor -> {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{'fp16' if a.fp16 else 'fp32'}), strict load lulus")


if __name__ == "__main__":
    main()
