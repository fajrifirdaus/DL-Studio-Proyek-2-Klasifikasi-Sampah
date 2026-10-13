"""promote.py — Uji kandidat, lalu promosikan ke submission/ HANYA kalau menang.

KENAPA ADA SKRIP INI?
---------------------
`submission/` harus selalu dalam keadaan BISA DINILAI. Kalau kandidat yang belum
teruji ditempel ke sana, submission bisa rusak dan skor putaran itu hilang.
Skrip ini memaksa urutannya: **uji dulu, promosikan belakangan.**

APA YANG DIPROMOSIKAN?
----------------------
`submission/` butuh DUA berkas yang WAJIB berpasangan:

    submission/model.py    KODE  - get_model() + get_transform()   (cetakan)
    submission/model.pth   BOBOT - state_dict hasil latihan        (hasil)

Skrip ini bisa mempromosikan KEDUANYA sekaligus, jadi alur "menang -> pindah ke
submission" selesai dalam satu langkah.

ALUR PAKAI
----------
    # 1) latih kandidat -> bobotnya OTOMATIS ke runs/<tag>.pth (submission aman)
    python train.py --data "<...>/train" --tag coba_resnet34 \
                    --arsitektur experiments/resnet34_model.py --backbone resnet34
    #    (kandidat lain: experiments/efficientnet_v2s_model.py + backbone efficientnet_v2_s,
    #     tetapi resolusi 384 -> tambahkan --batch-size 16)

    # 2) uji saja, tidak mengubah apa pun
    python experiments/promote.py --candidate runs/coba_resnet34.pth \
                                  --arsitektur experiments/resnet34_model.py --dry-run

    # 3) kalau menang, promosikan (model.py + model.pth berpasangan)
    python experiments/promote.py --candidate runs/coba_resnet34.pth \
                                  --arsitektur experiments/resnet34_model.py --yes

YANG DILAKUKAN SAAT PROMOSI
---------------------------
  1. SKOR LAMA & SKOR BARU diukur dengan `evaluate.py` RESMI (angka dari JSON-nya,
     bukan tebakan dari teks)
  2. Kalau kandidat lebih rendah -> promosi DIBATALKAN (paksa dengan --force)
  3. model.py & model.pth lama dicadangkan ke runs/_arsip/<cap>/
  4. model.py & model.pth kandidat disalin ke submission/
  5. DIPERIKSA ULANG: submission/ baru harus bisa dimuat (load_state_dict
     strict=True). Kalau gagal, otomatis DIKEMBALIKAN ke versi lama.
  6. Riwayat ditulis ke runs/_arsip/riwayat.jsonl

SKOR DIUKUR DI MANA?
    Folder `my_val/` (dibuat `train.py`) - pembagian data dihitung dari
    `seed = 42` yang tetap, jadi sama di semua run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================
SUB_DIR = Path("submission")
SUB_MODEL_PY = SUB_DIR / "model.py"
SUB_WEIGHTS = SUB_DIR / "model.pth"
VAL_DIR = Path("my_val")
ARKIP = Path("runs/_arsip")
EVALUATE = Path("evaluate.py")


# ============================================================
# MENGUKUR SKOR
# ============================================================
def ukur_skor(folder_sub: Path, data: Path) -> dict | None:
    """Jalankan evaluate.py RESMI pada folder submission tertentu.

    Angkanya diambil dari berkas JSON (`--out`), bukan dari teks keluaran -
    jauh lebih andal. Kembalikan None kalau gagal.
    """
    if not EVALUATE.is_file():
        print("  [GAGAL] evaluate.py tidak ada di root repo.")
        return None
    if not data.is_dir():
        print(f"  [GAGAL] folder validasi tidak ada: {data}")
        print("          Buat dulu dengan melatih sekali:  python train.py --data <folder train/>")
        return None
    if not (folder_sub / "model.pth").is_file():
        print(f"  [lewat] {folder_sub}/model.pth belum ada")
        return None

    with tempfile.TemporaryDirectory(prefix="promote_skor_") as tmp:
        keluar_json = Path(tmp) / "skor.json"
        hasil = subprocess.run(
            [sys.executable, str(EVALUATE), "--submission", str(folder_sub),
             "--data", str(data), "--out", str(keluar_json)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

        if keluar_json.is_file():
            try:
                return json.loads(keluar_json.read_text(encoding="utf-8"))
            except Exception:
                pass

        gabung = ((hasil.stdout or "") + (hasil.stderr or "")).strip()
        print("  keluaran evaluate.py:")
        for baris in gabung.splitlines()[-12:]:
            print("    " + baris)
        return None


def ringkas_skor(m: dict | None) -> str:
    if not m:
        return "-"
    return "macro-F1 {:.4f} | akurasi {:.4f} | {} gambar | {}s".format(
        m.get("macro_f1", 0.0), m.get("accuracy", 0.0),
        m.get("n_images", 0), m.get("seconds", 0))


# ============================================================
# PROGRAM UTAMA
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Uji kandidat, lalu promosikan ke submission/ hanya kalau menang")
    ap.add_argument("--candidate", required=True,
                    help="berkas bobot kandidat (mis. runs/coba_resnet34.pth)")
    ap.add_argument("--arsitektur", default=None,
                    help="berkas .py kandidat di experiments/ (get_model+get_transform). "
                         "WAJIB kalau arsitekturnya beda dari submission/model.py sekarang")
    ap.add_argument("--data", default=str(VAL_DIR), help="folder validasi (default my_val/)")
    ap.add_argument("--yes", action="store_true", help="benar-benar promosikan")
    ap.add_argument("--dry-run", action="store_true", help="hanya uji, jangan ubah apa pun")
    ap.add_argument("--force", action="store_true", help="promosikan walau skornya lebih rendah")
    a = ap.parse_args()

    kandidat = Path(a.candidate)
    if not kandidat.is_file():
        raise SystemExit(
            f"[GAGAL] berkas kandidat tidak ada: {kandidat}\n"
            "  Latih dulu:  python train.py --data <folder train/> --tag <nama>")
    if not a.yes and not a.dry_run:
        raise SystemExit(
            "[BERHENTI] Tambahkan --dry-run (uji saja) atau --yes (promosikan).\n"
            "  Tanpa salah satu itu, skrip ini tidak melakukan apa-apa.")
    arsitektur = Path(a.arsitektur) if a.arsitektur else None
    if arsitektur and not arsitektur.is_file():
        raise SystemExit(f"[GAGAL] berkas arsitektur tidak ada: {arsitektur}")

    print("=" * 70)
    print("  UJI KANDIDAT SEBELUM PROMOSI")
    print("=" * 70)
    print(f"  kandidat bobot : {kandidat}  ({kandidat.stat().st_size / 1e6:.1f} MB)")
    print(f"  kandidat kode  : {arsitektur if arsitektur else '(pakai submission/model.py sekarang)'}")
    print(f"  validasi       : {a.data}")
    print(f"  submission     : {SUB_DIR}/  (model.py {'ADA' if SUB_MODEL_PY.is_file() else 'TIDAK ADA'}"
          f", bobot {'ADA' if SUB_WEIGHTS.is_file() else 'TIDAK ADA'})")

    # ---------- [1/2] skor kandidat ----------
    # model.py kandidat = berkas experiments/ (kalau diberikan), kalau tidak
    # berarti arsitekturnya sama dengan submission/model.py yang sekarang.
    print("\n[1/2] SKOR KANDIDAT")
    with tempfile.TemporaryDirectory(prefix="promote_kandidat_") as tmp:
        folder_kandidat = Path(tmp) / "kandidat"
        folder_kandidat.mkdir()
        sumber_py = arsitektur if arsitektur else SUB_MODEL_PY
        if not sumber_py.is_file():
            raise SystemExit(
                f"[GAGAL] tidak ada model.py untuk diuji: {sumber_py}\n"
                "  Kalau arsitektur kandidat beda, sertakan --arsitektur experiments/....py")
        shutil.copy2(sumber_py, folder_kandidat / "model.py")
        shutil.copy2(kandidat, folder_kandidat / "model.pth")
        skor_baru = ukur_skor(folder_kandidat, Path(a.data))

    print(f"      {ringkas_skor(skor_baru)}")

    # ---------- [2/2] skor submission sekarang ----------
    print("\n[2/2] SKOR SUBMISSION SEKARANG")
    skor_lama = ukur_skor(SUB_DIR, Path(a.data)) if SUB_WEIGHTS.is_file() else None
    print(f"      {ringkas_skor(skor_lama)}")

    # ---------- perbandingan ----------
    print("\n" + "-" * 70)
    f_baru = (skor_baru or {}).get("macro_f1")
    f_lama = (skor_lama or {}).get("macro_f1")
    print(f"  macro-F1 sekarang (submission) : {f_lama:.4f}" if f_lama is not None
          else "  macro-F1 sekarang (submission) : -  (belum ada)")
    print(f"  macro-F1 kandidat              : {f_baru:.4f}" if f_baru is not None
          else "  macro-F1 kandidat              : -  (gagal diukur)")

    menang = None
    if f_baru is None:
        print("\n  PERINGATAN: skor kandidat TIDAK TERBACA - promosi tidak dijalankan.")
        print("  Penyebab tersering: model.py dan model.pth kandidat TIDAK berpasangan")
        print("  (mis. model.py ResNet-34 tetapi bobotnya ResNet-18).")
        print("  Periksa keluaran evaluate.py di atas, lalu pastikan keduanya dari")
        print("  percobaan yang SAMA (tag yang sama).")
        print("\n  submission/ TIDAK diubah (masih versi lama).")
        if a.dry_run:
            print("[dry-run] benar-benar tidak ada yang ditulis.")
        return
    if f_lama is None:
        print("  (belum ada pembanding - kandidat ini akan jadi submission pertama)")
        menang = True
    else:
        selisih = f_baru - f_lama
        tanda = "NAIK" if selisih > 0 else ("SERI" if selisih == 0 else "TURUN")
        print(f"  selisih                        : {selisih:+.4f}  ({tanda})")
        n_gambar = (skor_lama or skor_baru or {}).get("n_images", 0)
        if n_gambar:
            # Perkiraan kasar: dengan n gambar, selisih di bawah angka ini bisa saja
            # cuma keberuntungan. Ini BUKAN uji statistik formal - hanya pengingat
            # supaya tidak menyimpulkan terlalu jauh dari selisih kecil.
            ambang = 1.96 * 0.5 / (n_gambar ** 0.5)
            print(f"  catatan: pada {n_gambar} gambar, selisih di bawah +-{ambang:.3f} "
                  "bisa cuma keberuntungan (perkiraan)")
        menang = selisih > 0
    print("-" * 70)

    if not menang and not a.force:
        print("\n  -> Kandidat TIDAK lebih baik. Promosi DIBATALKAN.")
        print("     (paksa dengan --force kalau memang mau menempelkannya)")
        if a.dry_run:
            print("\n[dry-run] tidak ada yang diubah.")
        return

    # ---------- ringkasan rencana ----------
    print("\nRENCANA PROMOSI (model.py + model.pth jadi berpasangan):")
    if arsitektur:
        print(f"  + model.py  : {arsitektur}  ->  {SUB_MODEL_PY}   (KODE DIGANTI)")
    else:
        print(f"  + model.py  : tidak diubah (arsitektur sama)")
    print(f"  + model.pth : {kandidat}  ->  {SUB_WEIGHTS}")

    if a.dry_run:
        print("\n[dry-run] tidak ada yang diubah. Jalankan ulang dengan --yes untuk mempromosikan.")
        return

    # ---------- cadangkan versi lama ----------
    cap = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_arsip = ARKIP / cap
    folder_arsip.mkdir(parents=True, exist_ok=True)
    cadangan_py = None
    if SUB_MODEL_PY.is_file():
        cadangan_py = folder_arsip / "model.py"
        shutil.copy2(SUB_MODEL_PY, cadangan_py)
    if SUB_WEIGHTS.is_file():
        shutil.copy2(SUB_WEIGHTS, folder_arsip / "model.pth")
    print(f"\n[arsip] versi lama -> {folder_arsip}/")

    # ---------- promosikan ----------
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    if arsitektur:
        shutil.copy2(arsitektur, SUB_MODEL_PY)
        print(f"[promosi] model.py  : {arsitektur} -> {SUB_MODEL_PY}")
    shutil.copy2(kandidat, SUB_WEIGHTS)
    print(f"[promosi] model.pth : {kandidat} -> {SUB_WEIGHTS}")

    # ---------- PERIKSA ULANG: submission baru harus bisa dimuat ----------
    print("\n[periksa] uji muat submission baru (load_state_dict strict=True) ...")
    skor_cek = ukur_skor(SUB_DIR, Path(a.data))
    if skor_cek is None:
        print("  [GAGAL] submission baru TIDAK bisa dimuat! Mengembalikan versi lama ...")
        if cadangan_py:
            shutil.copy2(cadangan_py, SUB_MODEL_PY)
        cadangan_pth = folder_arsip / "model.pth"
        if cadangan_pth.is_file():
            shutil.copy2(cadangan_pth, SUB_WEIGHTS)
        else:
            SUB_WEIGHTS.unlink(missing_ok=True)
        raise SystemExit(
            "[DIBATALKAN] submission dikembalikan ke versi lama.\n"
            "  Penyebab paling sering: model.py dan model.pth TIDAK berpasangan\n"
            "  (mis. model.py ResNet-34 tapi bobotnya ResNet-18).\n"
            "  Pastikan keduanya dari percobaan yang SAMA.")

    print(f"  OK - {ringkas_skor(skor_cek)}")

    # ---------- catat riwayat ----------
    info = {"waktu": datetime.now().isoformat(timespec="seconds"),
            "kandidat_bobot": str(kandidat),
            "kandidat_kode": str(arsitektur) if arsitektur else "(tidak diubah)",
            "arsip": str(folder_arsip),
            "macro_f1_lama": f_lama, "macro_f1_baru": f_baru,
            "macro_f1_terpasang": skor_cek.get("macro_f1")}
    ARKIP.mkdir(parents=True, exist_ok=True)
    with (ARKIP / "riwayat.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(info) + "\n")

    print("\n" + "=" * 70)
    print("  SELESAI - submission/ sudah diperbarui")
    print("=" * 70)
    print(f"  macro-F1 terpasang : {skor_cek.get('macro_f1'):.4f}")
    print(f"  riwayat            : {ARKIP / 'riwayat.jsonl'}")
    print("\n  LANGKAH WAJIB BERIKUTNYA:")
    print("    1. commit DUA berkas sekaligus: submission/model.py + submission/model.pth")
    print("    2. uji sekali lagi dari laptop/Colab:")
    print(f"       python evaluate.py --submission ./submission --data {a.data}")


if __name__ == "__main__":
    main()
