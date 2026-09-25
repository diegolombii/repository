import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np

# ============================================================
# COSTANTI (raramente cambiano - editare qui se serve davvero)
# ============================================================

CLASSES_SANE = {"normal", "ls1", "ls2", "ls3", "ls4"}
SEQ_LEN = 7800
FRAME_LEN = 600
FRAME_HOP = 600
CALIB_N_SAMPLES = 200
CALIB_SEED = 42
QUANT_SCALE = 0.01210999395698309 #preso da da AI/App/network_details.h, tensore "input_output"
QUANT_ZERO_POINT = 0

RE_1 = re.compile(r"^(HIF)_(.+?)_(Voltage|Current)_(\d+)\.mat$", re.IGNORECASE)
RE_0_NORMAL = re.compile(r"^(voltage|current)_(normal)_(\d+)\.mat$", re.IGNORECASE)
RE_0_LS = re.compile(r"^(.+?)_(Voltage|Current)_(\d+)\.mat$", re.IGNORECASE)

@dataclass
class MatPair:
    class_type: str
    idx: int
    v_path: str
    i_path: str


# ============================================================
# FILE .MAT
# ============================================================

def list_mat_files(root: str) -> List[str]:
    out = []
    for r, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(".mat"):
                out.append(os.path.join(r, f))
    return sorted(out)


def parse_file_info(path: str) -> Tuple[str, str, int]:
    base = os.path.basename(path)
    m = RE_1.match(base)
    if m:
        return m.group(2).lower(), m.group(3).lower(), int(m.group(4))
    m = RE_0_NORMAL.match(base)
    if m:
        return m.group(2).lower(), m.group(1).lower(), int(m.group(3))
    m = RE_0_LS.match(base)
    if m:
        return m.group(1).lower(), m.group(2).lower(), int(m.group(3))
    raise ValueError(f"Nome file non riconosciuto: {base}")


def build_pairs(data_dir: str) -> List[MatPair]:
    files = list_mat_files(data_dir)
    if not files:
        raise RuntimeError(f"Nessun file .mat trovato in: {data_dir}")
    tmp = {}
    for path in files:
        try:
            cls, kind, idx = parse_file_info(path)
        except Exception:
            continue
        key = (cls, idx)
        if key not in tmp:
            tmp[key] = {"class_type": cls, "idx": idx, "voltage": None, "current": None}
        tmp[key][kind] = path
    pairs = [
        MatPair(item["class_type"], item["idx"], item["voltage"], item["current"])
        for item in tmp.values()
        if item["voltage"] is not None and item["current"] is not None
    ]
    pairs = sorted(pairs, key=lambda p: (p.class_type, p.idx))
    if not pairs:
        raise RuntimeError("Nessuna coppia completa tensione-corrente trovata.")
    return pairs


def load_matrix(path: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    with h5py.File(path, "r") as f:
        arr = None
        if "#refs#" not in f:
            raise ValueError(f"Gruppo #refs# non trovato in: {path}")
        for name in f["#refs#"]:
            obj = f["#refs#"][name]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and 10001 in obj.shape and 3 in obj.shape:
                arr = np.squeeze(obj[()])
                if arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                if arr.ndim != 2:
                    raise ValueError(f"Attesa matrice 2D, ottenuta {arr.shape}")
                break
        if arr is None:
            raise ValueError(f"Matrice non trovata nel file: {path}")
    T, _ = arr.shape
    if T > seq_len:
        arr = arr[-seq_len:, :]  # ultimi seq_len campioni, come inferenza_pc_onnx.py
    elif T < seq_len:
        raise ValueError(f"Attese almeno {seq_len} righe, trovate {T}")
    return arr.astype(np.float32)


# ============================================================
# SCALER E PREPROCESSING
# ============================================================

def load_scaler(scaler_path: str):
    with np.load(scaler_path) as data:
        return data["v_mean"], data["v_std"], data["i_mean"], data["i_std"]


def preprocess_pair(pair: MatPair, scaler, channels: List[int], seq_len: int = SEQ_LEN) -> np.ndarray:
    """Z-score + selezione canali per una coppia V/I (stessa normalizzazione
    di inferenza_pc_onnx.py::process()). Canali: V_a,V_b,V_c,I_a,I_b,I_c."""
    if any(c < 0 or c > 5 for c in channels):
        raise ValueError(f"Canali non validi: {channels}. Ammessi: 0..5")
    v_mean, v_std, i_mean, i_std = scaler
    V = load_matrix(pair.v_path, seq_len=seq_len)
    I = load_matrix(pair.i_path, seq_len=seq_len)
    VI = np.concatenate([(V - v_mean) / (v_std + 1e-12), (I - i_mean) / (i_std + 1e-12)], axis=1)
    return VI[:, channels].astype(np.float32)


# ============================================================
# 1. CALIBRAZIONE PER LA QUANTIZZAZIONE (solo file sani)
# ============================================================

def export_calibration_dataset(pairs: List[MatPair], scaler_path: str, out_npz: str, channels: List[int]):
    healthy = [p for p in pairs if p.class_type.lower() in CLASSES_SANE]
    if not healthy:
        raise RuntimeError("Nessun file sano trovato per la calibrazione.")
    print(f"File sani per calibrazione: {len(healthy)}/{len(pairs)}")

    scaler = load_scaler(scaler_path)
    starts = range(0, SEQ_LEN - FRAME_LEN + 1, FRAME_HOP)

    all_frames = []
    for pair in healthy:
        X = preprocess_pair(pair, scaler, channels)
        all_frames.extend(X[s:s + FRAME_LEN, :] for s in starts)
    all_frames = np.stack(all_frames, axis=0)

    rng = np.random.default_rng(CALIB_SEED)
    n_take = min(CALIB_N_SAMPLES, all_frames.shape[0])
    idx = rng.choice(all_frames.shape[0], size=n_take, replace=False)
    frames = all_frames[idx][:, np.newaxis, :, :].astype(np.float32)  # (N,1,600,n_ch)

    np.savez(out_npz, input=frames)
    print(f"Calibrazione -> {out_npz}  shape={frames.shape}  "
          f"range=[{frames.min():.3f}, {frames.max():.3f}]")


# ============================================================
# 2. DATASET DI TEST PER LA BOARD: int8 quantizzato E float32
# ============================================================

def quantize_int8(x_float: np.ndarray, scale: float = QUANT_SCALE, zero_point: int = QUANT_ZERO_POINT) -> np.ndarray:
    return np.clip(np.round(x_float / scale) + zero_point, -128, 127).astype(np.int8)


def export_device_datasets(pairs: List[MatPair], scaler_path: str, out_dataset_int8: str, out_dataset_float32: str, out_labels: str, channels: List[int]):
    scaler = load_scaler(scaler_path)
    labels = np.zeros(len(pairs), dtype=np.uint8)

    with open(out_dataset_int8, "wb") as f_int8, open(out_dataset_float32, "wb") as f_f32:
        for i, pair in enumerate(pairs):
            X = preprocess_pair(pair, scaler, channels)
            f_int8.write(quantize_int8(X).tobytes())       
            f_f32.write(X.astype(np.float32).tobytes())    
            labels[i] = 0 if pair.class_type.lower() in CLASSES_SANE else 1
            if (i + 1) % 200 == 0 or (i + 1) == len(pairs):
                print(f"  {i+1}/{len(pairs)} file processati")

    labels.tofile(out_labels)
    n_ch = len(channels)
    print(f"Dataset int8    -> {out_dataset_int8} ({len(pairs) * SEQ_LEN * n_ch} byte)")
    print(f"Dataset float32 -> {out_dataset_float32} ({len(pairs) * SEQ_LEN * n_ch * 4} byte)")
    print(f"Label           -> {out_labels} ({len(pairs)} byte, "
          f"sani={int((labels==0).sum())} HIF={int((labels==1).sum())})")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="C:/Users/Diego/TIROCINIO/modello onnx/hif_validation_test", help="Cartella con tutti i file di test (sani + HIF).")
    parser.add_argument("--calibration_dir", default=None, help="Cartella solo-sani per calibrare, se diversa da --data_dir.")
    parser.add_argument("--scaler", default="C:/Users/Diego/TIROCINIO/modello onnx/scaler_vi_global.npz")
    parser.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()

    n_ch = len(args.channels)
    out_calib = f"calibration_sani_ch{n_ch}.npz"
    out_dataset_int8 = f"dataset_int8_ch{n_ch}.bin"
    out_dataset_float32 = f"dataset_float32_ch{n_ch}.bin"
    out_labels = f"labels_uint8_ch{n_ch}.bin"

    print("=== 1/2 Calibrazione ===")
    export_calibration_dataset(
        build_pairs(args.calibration_dir or args.data_dir), args.scaler, out_calib, args.channels
    )

    print("\n=== 2/2 Dataset board (int8 + float32) ===")
    export_device_datasets(
        build_pairs(args.data_dir), args.scaler, out_dataset_int8, out_dataset_float32, out_labels, args.channels
    )

    print(f"\nFatto: {out_calib}, {out_dataset_int8}, {out_dataset_float32}, {out_labels}")


if __name__ == "__main__":
    main()