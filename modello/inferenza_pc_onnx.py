import os
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import h5py
import numpy as np
import onnxruntime as onn
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


CLASSES_SANE = ["normal", "ls1", "ls2", "ls3", "ls4"]

RE_HIF = re.compile(r"^(HIF)_(.+?)_(Voltage|Current)_(\d+)\.mat$", re.IGNORECASE)
RE_NORMAL = re.compile(r"^(voltage|current)_(normal)_(\d+)\.mat$", re.IGNORECASE)
RE_LS = re.compile(r"^(.+?)_(Voltage|Current)_(\d+)\.mat$", re.IGNORECASE)


@dataclass
class MatPair:
    class_type: str
    idx: int
    v_path: str
    i_path: str


def list_mat_files(root: str) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".mat"):
                out.append(os.path.join(r, f))
    return sorted(out)


def load_scaler():
    path = "C:/Users/Diego/TIROCINIO/modello onnx/scaler_vi_global.npz"
    with np.load(path) as data:
        return data["v_mean"], data["v_std"], data["i_mean"], data["i_std"]


def parse_file_info(path: str) -> Tuple[str, str, int]:
    base = os.path.basename(path)

    m = RE_HIF.match(base)
    if m:
        return m.group(2).lower(), m.group(3).lower(), int(m.group(4))

    m = RE_NORMAL.match(base)
    if m:
        return m.group(2).lower(), m.group(1).lower(), int(m.group(3))

    m = RE_LS.match(base)
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
        except ValueError:
            continue

        key = (cls, idx)

        if key not in tmp:
            tmp[key] = {
                "class_type": cls,
                "idx": idx,
                "voltage": None,
                "current": None,
            }

        tmp[key][kind] = path

    pairs = [
        MatPair(
            class_type=item["class_type"],
            idx=item["idx"],
            v_path=item["voltage"],
            i_path=item["current"],
        )
        for item in tmp.values()
        if item["voltage"] is not None and item["current"] is not None
    ]

    pairs = sorted(pairs, key=lambda p: (p.class_type, p.idx))

    if not pairs:
        raise RuntimeError("Nessuna coppia completa tensione/corrente trovata.")

    return pairs


def load_matrix(path: str, seq_len: int = 7800) -> np.ndarray:
    with h5py.File(path, "r") as f:
        arr = None

        for name in f["#refs#"]:
            obj = f["#refs#"][name]

            if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                if 10001 in obj.shape and 3 in obj.shape:
                    arr = np.squeeze(obj[()])

                    if arr.shape[0] < arr.shape[1]:
                        arr = arr.T

                    break

        if arr is None:
            raise ValueError(f"Matrice non trovata in: {path}")

    if arr.shape[0] < seq_len:
        raise ValueError(f"Attese almeno {seq_len} righe, trovate {arr.shape[0]}")

    return arr[-seq_len:, :].astype(np.float32)


def process(V, I, v_mean, v_std, i_mean, i_std):
    V_scaled = (V - v_mean) / (v_std + 1e-12)
    I_scaled = (I - i_mean) / (i_std + 1e-12)
    return np.concatenate([V_scaled, I_scaled], axis=1)


class FrameDataset:
    def __init__(self, pairs: List[MatPair], scaler):
        self.pairs = pairs
        self.v_mean, self.v_std, self.i_mean, self.i_std = scaler

        self.y_by_file = []
        self.X_by_file: Optional[List[np.ndarray]] = []
        self.frame_index: List[Tuple[int, int]] = []

        starts = list(range(0, 7800, 600))

        for file_id, pair in enumerate(self.pairs):
            self.y_by_file.append(
                0 if pair.class_type in CLASSES_SANE else 1
            )

            self.X_by_file.append(self._load_full(pair))

            for start in starts:
                self.frame_index.append((file_id, start))

    def _load_full(self, pair: MatPair):
        V = load_matrix(pair.v_path)
        I = load_matrix(pair.i_path)

        return process(V, I, self.v_mean, self.v_std, self.i_mean, self.i_std,)
    
    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx: int):
        file_id, start = self.frame_index[idx]

        Xfull = self.X_by_file[file_id]
        frame = Xfull[start:start + 600, :]

        return frame.astype(np.float32), np.int64(file_id)


def batch_recon_error(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.abs(x_hat - x).mean(axis=(1, 2))


def collect_frame_errors(model, dl: DataLoader):
    errs = []
    fids = []
    times = []

    out_min = np.inf
    out_max = -np.inf

    for X, fid in dl:
        X = X.numpy()

        t0 = time.perf_counter()
        x_hat = model.run(["reconstruction"], {"input": X})[0]
        t1 = time.perf_counter()

        times.append(t1 - t0)

        out_min = min(out_min, float(x_hat.min()))
        out_max = max(out_max, float(x_hat.max()))

        errs.append(batch_recon_error(x_hat, X))
        fids.append(fid.numpy().astype(np.int64))

    return (np.concatenate(errs), np.concatenate(fids), np.array(times), out_min, out_max, )


def aggregate_file_scores(frame_scores, file_ids, n_files):
    scores = np.full(n_files, np.nan, dtype=np.float64)

    for file_id in range(n_files):
        s = frame_scores[file_ids == file_id]

        if s.size > 0:
            scores[file_id] = float(np.quantile(s, 0.95))

    return scores


def load_threshold(path: str, key: str) -> float:
    with open(path, "r") as f:
        for line in f:
            if line.startswith(key + "="):
                return float(line.strip().split("=", 1)[1])

    raise RuntimeError(f"Chiave '{key}' non trovata in {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_type", choices=["originale", "quant"], default="quant", )
    parser.add_argument("--fp32_model", default="tcn_hif_autoencoder.onnx",)
    parser.add_argument("--quant_model", default="tcn_hif_autoencoder_quant_sani.onnx",)
    parser.add_argument("--threshold_path", default="threshold.txt",)
    parser.add_argument("--data_dir", default="C:/Users/Diego/TIROCINIO/modello onnx/hif_validation",)
    
    args = parser.parse_args()

    if args.model_type == "originale":
        onnx_path = args.fp32_model
        threshold_key = "threshold_file"
        suffix = "originale"
    else:
        onnx_path = args.quant_model
        threshold_key = "threshold_file_quant"
        suffix = "quant"

    output_dir = (
        f"C:/Users/Diego/TIROCINIO/modello onnx/"
        f"inferenza pc batch {args.batch_size} onnx {suffix}"
    )

    os.makedirs(output_dir, exist_ok=True)

    test_pairs = build_pairs(args.data_dir)
    print("Numero file:", len(test_pairs))

    scaler = load_scaler()
    ds_test = FrameDataset(test_pairs, scaler)

    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"Modello usato ({args.model_type}): {onnx_path}")

    model = onn.InferenceSession(onnx_path)

    (
        test_frame_err,
        test_file_ids,
        inference_time,
        out_min,
        out_max,
    ) = collect_frame_errors(model, dl_test)

    print("Range output rete:")
    print("out_min:", out_min)
    print("out_max:", out_max)

    pd.DataFrame({
        "frame_err": test_frame_err,
        "file_id": test_file_ids,
    }).to_csv(
        os.path.join(output_dir, "test_frame_errors.csv"),
        index=False,
    )

    pd.DataFrame({
        "batch_id": np.arange(len(inference_time)),
        "inference_time": inference_time,
    }).to_csv(
        os.path.join(output_dir, "inference_time.csv"),
        index=False,
    )

    test_file_scores = aggregate_file_scores(
        test_frame_err,
        test_file_ids,
        len(test_pairs),
    )

    threshold = load_threshold(
        args.threshold_path,
        threshold_key,
    )

    print(f"Soglia usata ({threshold_key}): {threshold}")

    y_pred_file = (test_file_scores > threshold).astype(np.int64)

    cm = confusion_matrix(
        ds_test.y_by_file,
        y_pred_file,
        labels=[0, 1],
    )

    print("Confusion matrix:")
    print(cm)

    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(
        os.path.join(output_dir, "confusion_test_binary.csv")
    )

    pd.DataFrame({
        "file": [
            f"{p.class_type}_{p.idx}"
            for p in test_pairs
        ],
        "label_vera": ds_test.y_by_file,
        "score": test_file_scores,
        "label_predetta": y_pred_file,
    }).to_csv(
        os.path.join(output_dir, "test_file_scores.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()