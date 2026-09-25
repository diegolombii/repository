import os
import re
import time
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import tensorflow.lite as tflite
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


def build_pairs(data_dir):
    tmp = {}

    for root, _, files in os.walk(data_dir):
        for name in files:
            if not name.lower().endswith(".mat"):
                continue

            path = os.path.join(root, name)

            m = RE_HIF.match(name)
            if m:
                cls, kind, idx = m.group(2).lower(), m.group(3).lower(), int(m.group(4))
            else:
                m = RE_NORMAL.match(name)
                if m:
                    cls, kind, idx = m.group(2).lower(), m.group(1).lower(), int(m.group(3))
                else:
                    m = RE_LS.match(name)
                    if not m:
                        continue
                    cls, kind, idx = m.group(1).lower(), m.group(2).lower(), int(m.group(3))

            key = (cls, idx)
            tmp.setdefault(key, {"voltage": None, "current": None})
            tmp[key][kind] = path

    pairs = [
        MatPair(cls, idx, d["voltage"], d["current"])
        for (cls, idx), d in tmp.items()
        if d["voltage"] is not None and d["current"] is not None
    ]

    return sorted(pairs, key=lambda p: (p.class_type, p.idx))


def load_matrix(path, seq_len=7800):
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
        raise ValueError(f"Matrice non trovata: {path}")

    if arr.shape[0] < seq_len:
        raise ValueError(f"Troppi pochi campioni in {path}")

    return arr[-seq_len:, :].astype(np.float32)


def load_scaler(path):
    with np.load(path) as data:
        return data["v_mean"], data["v_std"], data["i_mean"], data["i_std"]


def process(V, I, scaler):
    v_mean, v_std, i_mean, i_std = scaler

    V = (V - v_mean) / (v_std + 1e-12)
    I = (I - i_mean) / (i_std + 1e-12)

    return np.concatenate([V, I], axis=1).astype(np.float32)


class FrameDataset:
    def __init__(self, pairs, scaler):
        self.pairs = pairs
        self.y_by_file = []
        self.frames = []

        for fid, pair in enumerate(pairs):
            self.y_by_file.append(0 if pair.class_type in CLASSES_SANE else 1)

            X = process(
                load_matrix(pair.v_path),
                load_matrix(pair.i_path),
                scaler,
            )

            for start in range(0, 7800, 600):
                self.frames.append((X[start:start + 600], fid))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame, fid = self.frames[idx]
        return frame.astype(np.float32), np.int64(fid)


def collect_frame_errors(interpreter, dl):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    errs = []
    fids = []
    times = []

    for X, fid in dl:
        X = X.numpy().astype(np.float32)          # (batch, 600, 6)
        X_tflite = np.transpose(X, (0, 2, 1))   # (batch, 6, 600)

        t0 = time.perf_counter()

        interpreter.set_tensor(input_index, X_tflite)
        interpreter.invoke()
        x_hat = interpreter.get_tensor(output_index)

        times.append(time.perf_counter() - t0)

        # L'output TFLite è già nel formato (batch, 600, 6),
        # quindi NON va trasposto.
        errs.append(np.abs(x_hat - X).mean(axis=(1, 2)))
        fids.append(fid.numpy())

    return np.concatenate(errs), np.concatenate(fids), np.array(times)


def aggregate_file_scores(frame_scores, file_ids, n_files):
    return np.array([
        np.quantile(frame_scores[file_ids == fid], 0.95)
        for fid in range(n_files)
    ])


def load_threshold(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("threshold_file="):
                return float(line.split("=", 1)[1])

    raise RuntimeError("threshold_file non trovata")


def main():
    DATA_DIR = "C:/Users/Diego/TIROCINIO/modello onnx/hif_validation"
    SCALER_PATH = "C:/Users/Diego/TIROCINIO/modello onnx/scaler_vi_global.npz"
    MODEL_PATH = "tcn_hif_autoencoder.tflite"
    THRESHOLD_PATH = "threshold.txt"
    OUTPUT_DIR = "C:/Users/Diego/TIROCINIO/modello onnx/inferenza pc tflite"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = build_pairs(DATA_DIR)
    scaler = load_scaler(SCALER_PATH)

    dataset = FrameDataset(pairs, scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print("Input TFLite:", interpreter.get_input_details()[0]["shape"])
    print("Output TFLite:", interpreter.get_output_details()[0]["shape"])

    frame_err, file_ids, times = collect_frame_errors(interpreter, dataloader)

    file_scores = aggregate_file_scores(
        frame_err,
        file_ids,
        len(pairs),
    )

    threshold = load_threshold(THRESHOLD_PATH)
    y_pred = (file_scores > threshold).astype(np.int64)

    cm = confusion_matrix(dataset.y_by_file, y_pred, labels=[0, 1])

    print("Soglia:", threshold)
    print("Confusion matrix:")
    print(cm)

    pd.DataFrame({
        "file": [f"{p.class_type}_{p.idx}" for p in pairs],
        "label_vera": dataset.y_by_file,
        "score": file_scores,
        "label_predetta": y_pred,
    }).to_csv(
        os.path.join(OUTPUT_DIR, "test_file_scores.csv"),
        index=False,
    )

    pd.DataFrame({
        "inference_time": times
    }).to_csv(
        os.path.join(OUTPUT_DIR, "inference_time.csv"),
        index=False,
    )

    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(
        os.path.join(OUTPUT_DIR, "confusion_test_binary.csv")
    )


if __name__ == "__main__":
    main()