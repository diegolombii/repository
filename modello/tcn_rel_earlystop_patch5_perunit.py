#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(PHYSICS-INFORMED + FILE-LEVEL SCORING / FILE-LEVEL THRESHOLDING)

- physics loss is computed on de-normalized physical units
- frame-level ROC is explicitly treated as a weak-label diagnostic metric
- sequence cropping is configurable instead of always taking the tail
- file loading is more efficient through optional preload + LRU caching
- I/O Bottleneck fixed by forcing memory pre-loading.
- Inference DataLoaders are unshuffled for chronological temporal plotting.
- Missing channels are Zero-Padded to prevent fake symmetry.

Decision logic:
    frame errors -> one file score per file -> file threshold

Main behavior:
- Train only on non-fault files from --train_dir
- Compute reconstruction error per frame
- Aggregate frame errors into a single file anomaly score
  using quantile
- Fit threshold on TRAIN FILE SCORES only
  using percentile thresholding (log1p) + safety clamp
- Classify a file as faulty if:
      file_score > threshold

Outputs:
- Weak-label frame ROC (CSV + PNG) [diagnostic only]
- File ROC (CSV + PNG) based on file scores [primary metric]
- Final classification report + confusion matrix
"""

from __future__ import annotations

import os
import re
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt

# ============================================================
# CUDA PERFORMANCE SETTINGS
# ============================================================

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# ============================================================
# Utility
# ============================================================

NONFAULT_CLASSES = ["normal", "LS1", "LS2", "LS3", "LS4"]
NONFAULT_LOWER = {c.lower() for c in NONFAULT_CLASSES}


def is_nonfault_class(cls: str) -> bool:
    return cls.strip().lower() in NONFAULT_LOWER


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def split_train_monitor_pairs(
    pairs: List["MatPair"],
    monitor_frac: float,
    seed: int
) -> Tuple[List["MatPair"], List["MatPair"]]:
    """
    Split healthy files into train/monitor subsets while preserving class mix.
    The monitor subset is used only for early stopping, not for fitting weights.
    """
    if monitor_frac <= 0.0:
        return list(pairs), []

    rng = random.Random(seed)
    buckets: Dict[str, List["MatPair"]] = {}
    for p in pairs:
        buckets.setdefault(p.class_name.strip().lower(), []).append(p)

    train_pairs, monitor_pairs = [], []
    for _, bucket in buckets.items():
        bucket = list(bucket)
        rng.shuffle(bucket)
        n_mon = max(1, int(round(len(bucket) * monitor_frac))) if len(bucket) > 1 else 0
        n_mon = min(n_mon, max(0, len(bucket) - 1))
        monitor_pairs.extend(bucket[:n_mon])
        train_pairs.extend(bucket[n_mon:])

    if len(train_pairs) == 0:
        raise RuntimeError("monitor_frac is too large: no files left for weight training.")

    return sorted(train_pairs, key=lambda p: (p.class_name.lower(), p.idx)),            sorted(monitor_pairs, key=lambda p: (p.class_name.lower(), p.idx))


# ============================================================
# FILE PAIRING
# ============================================================

RE_A = re.compile(r"^(current|voltage)_(normal)_(\d+)\.mat$", re.IGNORECASE)
RE_A2 = re.compile(r"^(normal)_(current|voltage)_(\d+)\.mat$", re.IGNORECASE)
RE_B = re.compile(r"^(.+?)_(Voltage|Current)_(\d+)\.mat$", re.IGNORECASE)


@dataclass(frozen=True)
class MatPair:
    class_name: str
    idx: int
    v_path: str
    i_path: str


def list_mat_files(root: str) -> List[str]:
    out = []
    for r, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(".mat"):
                out.append(os.path.join(r, f))
    return sorted(out)


def parse_file_info(path: str) -> Tuple[str, str, int]:
    base = os.path.basename(path)
    m = RE_A.match(base)
    if m:
        return "normal", m.group(1).lower(), int(m.group(3))
    m = RE_A2.match(base)
    if m:
        return "normal", m.group(2).lower(), int(m.group(3))
    m = RE_B.match(base)
    if m:
        return m.group(1), m.group(2).lower(), int(m.group(3))
    raise ValueError(f"Unrecognized filename pattern: {base}")


def build_pairs(data_dir: str) -> List[MatPair]:
    files = list_mat_files(data_dir)
    if not files:
        raise RuntimeError(f"No .mat files found under: {data_dir}")

    v_map: Dict[Tuple[str, int], Tuple[str, str]] = {}
    i_map: Dict[Tuple[str, int], Tuple[str, str]] = {}

    for p in files:
        try:
            cls, kind, idx = parse_file_info(p)
        except Exception:
            continue
        key = (cls.strip().lower(), idx)
        if kind == "voltage":
            v_map[key] = (cls, p)
        else:
            i_map[key] = (cls, p)

    pairs: List[MatPair] = []
    for key in sorted(set(v_map.keys()) | set(i_map.keys())):
        if key not in v_map or key not in i_map:
            continue
        cls = v_map[key][0]
        pairs.append(MatPair(cls, key[1], v_map[key][1], i_map[key][1]))

    if not pairs:
        raise RuntimeError("No usable (V,I) pairs found.")
    return pairs


# ============================================================
# MAT LOADING
# ============================================================

def _best_numeric_array_from_h5(f: h5py.File) -> np.ndarray:
    best = None
    best_score = -1

    def visit(_name, obj):
        nonlocal best, best_score
        if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
            if obj.ndim in (2, 3):
                score = int(np.prod(obj.shape))
                if obj.ndim == 3 and (obj.shape[-1] == 1 or obj.shape[0] == 1):
                    score += 10_000_000
                if score > best_score:
                    best = obj
                    best_score = score

    f.visititems(visit)
    if best is None:
        raise ValueError("No numeric dataset found in HDF5 MAT.")
    return np.array(best)


def load_main_matrix(path: str) -> np.ndarray:
    try:
        md = loadmat(path)
        best = None
        best_size = -1
        for k, v in md.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                a = np.squeeze(v)
                if a.ndim == 2 and a.size > best_size:
                    best = a
                    best_size = a.size
        if best is None:
            raise ValueError("No numeric 2D array found.")
        arr = best
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr.astype(np.float32)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            arr = _best_numeric_array_from_h5(f)

        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            elif arr.shape[0] == 1:
                arr = arr[0, :, :]

        if arr.ndim != 2:
            raise ValueError("Expected 2D array.")
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr.astype(np.float32)


def ensure_n_channels(X: np.ndarray, n: int, strict: bool = False) -> np.ndarray:
    T, C = X.shape
    if C >= n:
        return X[:, :n]
    if strict:
        raise ValueError(
            f"Input has {C} channels but n_channels={n}. "
            "Set --allow_channel_duplication to enable fallback duplication."
        )
    #  Zero-pad missing phases instead of duplicating to avoid fake symmetry
    pad = np.zeros((T, n - C), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def crop_or_pad_sequence(X: np.ndarray,
                         target_len: int,
                         crop_mode: Literal["last", "first", "center", "random", "full"],
                         rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Crop or pad a sequence to target_len.
    """
    if target_len is None or target_len <= 0 or crop_mode == "full":
        return X

    T, C = X.shape
    if T == target_len:
        return X

    if T > target_len:
        if crop_mode == "last":
            start = T - target_len
        elif crop_mode == "first":
            start = 0
        elif crop_mode == "center":
            start = max(0, (T - target_len) // 2)
        elif crop_mode == "random":
            if rng is None:
                rng = np.random.RandomState(0)
            start = int(rng.randint(0, T - target_len + 1))
        else:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")
        return X[start:start + target_len, :]

    out = np.zeros((target_len, C), dtype=X.dtype)
    start = (target_len - T) // 2
    out[start:start + T, :] = X
    return out


# ============================================================
# FRAMING
# ============================================================

def frame_starts(seq_len: int, frame_len: int, hop: int) -> List[int]:
    if seq_len <= frame_len:
        return [0]
    return [k * hop for k in range(1 + (seq_len - frame_len) // hop)]


# ============================================================
# GLOBAL SCALER (TRAIN ONLY)
# ============================================================

class RunningStats:
    """Numerically stable per-channel running mean/variance (Welford)."""

    def __init__(self, n_channels: int):
        self.n = 0
        self.mean = np.zeros((n_channels,), dtype=np.float64)
        self.M2 = np.zeros((n_channels,), dtype=np.float64)

    def update_array(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expected 2D array for RunningStats.update_array")
        for row in X:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            raise ValueError("No samples seen by RunningStats")
        var = self.M2 / max(1, self.n)
        std = np.sqrt(np.maximum(var, 1e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


class GlobalVIScaler:
    def __init__(self, v_mean, v_std, i_mean, i_std):
        self.v_mean = v_mean.astype(np.float32)
        self.v_std = v_std.astype(np.float32)
        self.i_mean = i_mean.astype(np.float32)
        self.i_std = i_std.astype(np.float32)

    @staticmethod
    def fit(pairs: List[MatPair],
            seq_len: int,
            n_channels: int,
            crop_mode: str = "center",
            allow_channel_duplication: bool = False) -> "GlobalVIScaler":
        v_stats = RunningStats(n_channels)
        i_stats = RunningStats(n_channels)

        for p in pairs:
            V = ensure_n_channels(load_main_matrix(p.v_path), n_channels,
                                  strict=(not allow_channel_duplication))
            I = ensure_n_channels(load_main_matrix(p.i_path), n_channels,
                                  strict=(not allow_channel_duplication))
            V = crop_or_pad_sequence(V, seq_len, crop_mode)
            I = crop_or_pad_sequence(I, seq_len, crop_mode)
            v_stats.update_array(V)
            i_stats.update_array(I)

        v_mean, v_std = v_stats.finalize()
        i_mean, i_std = i_stats.finalize()
        return GlobalVIScaler(v_mean, v_std, i_mean, i_std)

    def transform(self, V: np.ndarray, I: np.ndarray, want_input: str) -> np.ndarray:
        parts = []
        if want_input in ("v", "vi"):
            parts.append((V - self.v_mean) / (self.v_std + 1e-12))
        if want_input in ("i", "vi"):
            parts.append((I - self.i_mean) / (self.i_std + 1e-12))
        if not parts:
            raise RuntimeError("No V/I parts selected.")
        return np.concatenate(parts, axis=1).astype(np.float32)

    def inv_stats_for_input(self, want_input: str) -> Tuple[np.ndarray, np.ndarray]:
        parts_mean = []
        parts_std = []
        if want_input in ("v", "vi"):
            parts_mean.append(self.v_mean)
            parts_std.append(self.v_std)
        if want_input in ("i", "vi"):
            parts_mean.append(self.i_mean)
            parts_std.append(self.i_std)
        if not parts_mean:
            raise RuntimeError("No V/I parts selected.")
        return np.concatenate(parts_mean), np.concatenate(parts_std)

    def save_npz(self, path: str, seq_len: int, n_channels: int, crop_mode: str):
        np.savez(
            path,
            v_mean=self.v_mean, v_std=self.v_std,
            i_mean=self.i_mean, i_std=self.i_std,
            seq_len=int(seq_len), n_channels=int(n_channels),
            crop_mode=str(crop_mode)
        )


# ============================================================
# Per-unit base estimation
# ============================================================

def estimate_vi_bases_from_pairs(
    pairs: List[MatPair],
    seq_len: int,
    n_channels: int,
    crop_mode: str = "center",
    allow_channel_duplication: bool = False,
) -> Tuple[float, float]:
    """
    Estimate voltage/current base values from healthy training data using
    average per-phase RMS. This is a fallback when explicit engineering
    base values are not provided.
    """
    v_rms_vals = []
    i_rms_vals = []

    for p in pairs:
        V = ensure_n_channels(
            load_main_matrix(p.v_path),
            n_channels,
            strict=(not allow_channel_duplication),
        )
        I = ensure_n_channels(
            load_main_matrix(p.i_path),
            n_channels,
            strict=(not allow_channel_duplication),
        )

        V = crop_or_pad_sequence(V, seq_len, crop_mode)
        I = crop_or_pad_sequence(I, seq_len, crop_mode)

        v_rms = np.sqrt(np.mean(V[:, :3] ** 2, axis=0))
        i_rms = np.sqrt(np.mean(I[:, :3] ** 2, axis=0))

        v_rms_vals.append(float(np.mean(v_rms)))
        i_rms_vals.append(float(np.mean(i_rms)))

    if len(v_rms_vals) == 0 or len(i_rms_vals) == 0:
        raise RuntimeError("Could not estimate per-unit bases from training pairs.")

    v_base = max(float(np.mean(v_rms_vals)), 1e-12)
    i_base = max(float(np.mean(i_rms_vals)), 1e-12)
    return v_base, i_base


# ============================================================
# DATASET
# ============================================================

class FrameDataset(Dataset):
    def __init__(self,
                 pairs: List[MatPair],
                 seq_len: int,
                 n_channels: int,
                 want_input: str,
                 scaler: GlobalVIScaler,
                 frame_len: int,
                 hop: int,
                 preload: bool = True,
                 crop_mode: str = "center",
                 allow_channel_duplication: bool = False,
                 max_cache_files: int = 64):

        self.pairs = pairs
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.want_input = want_input
        self.scaler = scaler
        self.frame_len = frame_len
        self.hop = hop
        self.crop_mode = crop_mode
        self.allow_channel_duplication = allow_channel_duplication
        self.max_cache_files = max(0, int(max_cache_files))
        self._rng = np.random.RandomState(12345)

        self.y_by_file = np.array(
            [0 if is_nonfault_class(p.class_name) else 1 for p in pairs],
            dtype=np.int64
        )

        self._X_by_file: Optional[List[np.ndarray]] = None
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        if preload:
            # Preload everything into RAM here to completely eliminate I/O Bottleneck
            self._X_by_file = [self._load_full(fid) for fid in range(len(pairs))]

        self.frame_index: List[Tuple[int, int]] = []
        self.frames_per_file = np.zeros((len(pairs),), dtype=np.int64)
        for fid in range(len(pairs)):
            Xfull = self._get_full_for_indexing(fid)
            starts = frame_starts(Xfull.shape[0], frame_len, hop)
            self.frames_per_file[fid] = len(starts)
            for s in starts:
                self.frame_index.append((fid, s))

    def __len__(self):
        return len(self.frame_index)

    def _prepare_pair(self, p: MatPair) -> np.ndarray:
        V = ensure_n_channels(load_main_matrix(p.v_path), self.n_channels,
                              strict=(not self.allow_channel_duplication))
        I = ensure_n_channels(load_main_matrix(p.i_path), self.n_channels,
                              strict=(not self.allow_channel_duplication))
        V = crop_or_pad_sequence(V, self.seq_len, self.crop_mode, self._rng)
        I = crop_or_pad_sequence(I, self.seq_len, self.crop_mode, self._rng)
        return self.scaler.transform(V, I, self.want_input)

    def _load_full(self, fid: int) -> np.ndarray:
        return self._prepare_pair(self.pairs[fid])

    def _get_full_for_indexing(self, fid: int) -> np.ndarray:
        if self._X_by_file is not None:
            return self._X_by_file[fid]
        X = self._cache.get(fid)
        if X is not None:
            self._cache.move_to_end(fid)
            return X
        X = self._load_full(fid)
        if self.max_cache_files > 0:
            self._cache[fid] = X
            self._cache.move_to_end(fid)
            while len(self._cache) > self.max_cache_files:
                self._cache.popitem(last=False)
        return X

    def __getitem__(self, idx: int):
        fid, start = self.frame_index[idx]
        Xfull = self._X_by_file[fid] if self._X_by_file is not None else self._get_full_for_indexing(fid)

        frame = Xfull[start:start + self.frame_len, :]
        if frame.shape[0] < self.frame_len:
            pad = np.zeros((self.frame_len - frame.shape[0], frame.shape[1]), dtype=frame.dtype)
            frame = np.vstack([frame, pad])

        y = int(self.y_by_file[fid])
        return (
            torch.from_numpy(frame).float(),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(fid, dtype=torch.long),
        )


# ============================================================
# TCN ARCHITECTURE
# ============================================================

class ResidualDilatedBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation, padding=pad)
        self.gn1 = nn.GroupNorm(min(8, ch), ch)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(ch, ch, kernel_size, dilation=dilation, padding=pad)
        self.gn2 = nn.GroupNorm(min(8, ch), ch)
        self.act2 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        r = x
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.drop(x)
        x = self.act2(self.gn2(self.conv2(x)))
        return x + r


class TCNAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_ch: int = 64, latent_ch: int = 16,
                 kernel_size: int = 5, dilations: Tuple[int, ...] = (1, 2, 4, 8),
                 dropout: float = 0.1, use_downsample: bool = True):
        super().__init__()
        self.use_downsample = use_downsample

        self.in_conv = nn.Sequential(
            nn.Conv1d(n_features, hidden_ch, kernel_size=1),
            nn.GroupNorm(min(8, hidden_ch), hidden_ch),
            nn.ReLU(inplace=True),
        )

        self.enc1 = nn.Sequential(*[
            ResidualDilatedBlock(hidden_ch, kernel_size, d, dropout) for d in dilations
        ])

        if use_downsample:
            self.down1 = nn.Sequential(
                nn.Conv1d(hidden_ch, hidden_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, hidden_ch), hidden_ch),
                nn.ReLU(inplace=True),
            )
            self.enc2 = nn.Sequential(*[
                ResidualDilatedBlock(hidden_ch, kernel_size, d, dropout) for d in dilations
            ])
            self.down2 = nn.Sequential(
                nn.Conv1d(hidden_ch, hidden_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, hidden_ch), hidden_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.down1 = nn.Identity()
            self.enc2 = nn.Identity()
            self.down2 = nn.Identity()

        self.to_latent = nn.Conv1d(hidden_ch, latent_ch, 1)
        self.from_latent = nn.Conv1d(latent_ch, hidden_ch, 1)

        if use_downsample:
            self.up1 = nn.Sequential(
                nn.ConvTranspose1d(hidden_ch, hidden_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, hidden_ch), hidden_ch),
                nn.ReLU(inplace=True),
            )
            self.dec1 = nn.Sequential(*[
                ResidualDilatedBlock(hidden_ch, kernel_size, d, dropout) for d in dilations
            ])
            self.up2 = nn.Sequential(
                nn.ConvTranspose1d(hidden_ch, hidden_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(8, hidden_ch), hidden_ch),
                nn.ReLU(inplace=True),
            )
            self.dec2 = nn.Sequential(*[
                ResidualDilatedBlock(hidden_ch, kernel_size, d, dropout) for d in dilations
            ])
        else:
            self.up1 = nn.Identity()
            self.dec1 = nn.Identity()
            self.up2 = nn.Identity()
            self.dec2 = nn.Identity()

        self.out_conv = nn.Conv1d(hidden_ch, n_features, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, L)
        h = self.in_conv(x)
        h = self.enc1(h)

        h = self.down1(h)
        h = self.enc2(h)
        h = self.down2(h)

        z = self.to_latent(h)
        h = self.from_latent(z)

        h = self.up1(h)
        h = self.dec1(h)
        h = self.up2(h)
        h = self.dec2(h)

        y = self.out_conv(h)
        return y.transpose(1, 2)  # (B, L, F)


# ============================================================
# Optional auxiliary losses
# ============================================================

def asymmetry_loss(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compare reconstruction error on positive vs negative samples.
    Each side is normalized by its own active sample count.
    """
    err = (x_hat - x).abs()
    pos_mask = (x > 0).float()
    neg_mask = (x < 0).float()

    reduce_dims = tuple(range(1, err.ndim))
    pos_sum = (err * pos_mask).sum(dim=reduce_dims)
    neg_sum = (err * neg_mask).sum(dim=reduce_dims)
    pos_cnt = pos_mask.sum(dim=reduce_dims).clamp_min(eps)
    neg_cnt = neg_mask.sum(dim=reduce_dims).clamp_min(eps)

    #  Divided by the actual count of pos/neg samples, not total tensor length
    pos_err = pos_sum / pos_cnt
    neg_err = neg_sum / neg_cnt
    return torch.abs(pos_err - neg_err).mean()


def frequency_loss(x_hat, x):
    x32 = x.float()
    x_hat32 = x_hat.float()
    X = torch.fft.rfft(x32, dim=1)
    X_hat = torch.fft.rfft(x_hat32, dim=1)
    return torch.mean(torch.abs(X - X_hat))


# ============================================================
# Physics-informed loss (grouped families)
# ============================================================

def physics_loss(x: torch.Tensor,
                 x_hat: torch.Tensor,
                 means: torch.Tensor,
                 stds: torch.Tensor,
                 v_base: float,
                 i_base: float,
                 w_zero: float = 0.0,
                 w_sym: float = 0.0,
                 w_power: float = 0.0,
                 w_energy: float = 0.0) -> torch.Tensor:
    """
    Physics-informed loss calculated on per-unit voltage/current for
    zero-sequence and symmetry terms.

    Note:
    - zero-sequence and symmetry are evaluated in per-unit quantities
    - power and energy are kept in physical units
    """
    if x_hat.shape[-1] < 6:
        return torch.tensor(0.0, device=x_hat.device)

    x_raw = x * stds + means
    x_hat_raw = x_hat * stds + means

    Va, Vb, Vc = x_hat_raw[:, :, 0], x_hat_raw[:, :, 1], x_hat_raw[:, :, 2]
    Ia, Ib, Ic = x_hat_raw[:, :, 3], x_hat_raw[:, :, 4], x_hat_raw[:, :, 5]

    loss = torch.tensor(0.0, device=x_hat.device)

    Va_pu, Vb_pu, Vc_pu = Va / v_base, Vb / v_base, Vc / v_base
    Ia_pu, Ib_pu, Ic_pu = Ia / i_base, Ib / i_base, Ic / i_base

    if w_zero > 0:
        L_3V0 = torch.mean((Va_pu + Vb_pu + Vc_pu) ** 2)
        L_3I0 = torch.mean((Ia_pu + Ib_pu + Ic_pu) ** 2)
        loss = loss + w_zero * (L_3V0 + L_3I0)

    if w_sym > 0:
        sym_V_signal = (Va_pu - Vb_pu) ** 2 + (Vb_pu - Vc_pu) ** 2 + (Vc_pu - Va_pu) ** 2
        sym_I_signal = (Ia_pu - Ib_pu) ** 2 + (Ib_pu - Ic_pu) ** 2 + (Ic_pu - Ia_pu) ** 2

        L_sym_V = torch.var(sym_V_signal, dim=1).mean()
        L_sym_I = torch.var(sym_I_signal, dim=1).mean()

        loss = loss + w_sym * (L_sym_V + L_sym_I)
    if w_power > 0:
        p = Va * Ia + Vb * Ib + Vc * Ic
        L_power = torch.var(p, dim=1).mean()
        loss = loss + w_power * L_power

    if w_energy > 0:
        e_x = torch.sum(x_raw ** 2, dim=(1, 2))
        e_xhat = torch.sum(x_hat_raw ** 2, dim=(1, 2))
        L_energy = torch.mean((e_x - e_xhat) ** 2)
        loss = loss + w_energy * L_energy

    return loss




# ============================================================
# Physics component diagnostics
# ============================================================

def physics_zero_seq_components(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
    v_base: float,
    i_base: float,
) -> Dict[str, torch.Tensor]:
    """
    Return separate zero-sequence voltage/current components computed
    in per-unit quantities.
    """
    z = torch.tensor(0.0, device=x_hat.device)

    if x_hat.shape[-1] < 6:
        return {
            "L_3V0_pu": z,
            "L_3I0_pu": z,
            "ratio_V0_to_I0_pu": z,
        }

    x_hat_raw = x_hat * stds + means

    Va, Vb, Vc = x_hat_raw[:, :, 0], x_hat_raw[:, :, 1], x_hat_raw[:, :, 2]
    Ia, Ib, Ic = x_hat_raw[:, :, 3], x_hat_raw[:, :, 4], x_hat_raw[:, :, 5]

    Va_pu, Vb_pu, Vc_pu = Va / v_base, Vb / v_base, Vc / v_base
    Ia_pu, Ib_pu, Ic_pu = Ia / i_base, Ib / i_base, Ic / i_base

    L_3V0_pu = torch.mean((Va_pu + Vb_pu + Vc_pu) ** 2)
    L_3I0_pu = torch.mean((Ia_pu + Ib_pu + Ic_pu) ** 2)

    ratio = L_3V0_pu / torch.clamp(L_3I0_pu, min=1e-12)

    return {
        "L_3V0_pu": L_3V0_pu,
        "L_3I0_pu": L_3I0_pu,
        "ratio_V0_to_I0_pu": ratio,
    }


@torch.no_grad()
def evaluate_zero_seq_components(
    model,
    loader,
    device,
    means: torch.Tensor,
    stds: torch.Tensor,
    v_base: float,
    i_base: float,
) -> Dict[str, float]:
    """
    Average per-unit zero-sequence physics components over a loader.
    """
    model.eval()

    vals_v = []
    vals_i = []
    vals_ratio = []

    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        x_hat = model(x)

        comps = physics_zero_seq_components(
            x, x_hat, means, stds, v_base=v_base, i_base=i_base
        )

        vals_v.append(float(comps["L_3V0_pu"].detach().item()))
        vals_i.append(float(comps["L_3I0_pu"].detach().item()))
        vals_ratio.append(float(comps["ratio_V0_to_I0_pu"].detach().item()))

    if len(vals_v) == 0:
        return {
            "avg_L_3V0_pu": np.nan,
            "avg_L_3I0_pu": np.nan,
            "avg_ratio_V0_to_I0_pu": np.nan,
            "n_batches": 0,
        }

    return {
        "avg_L_3V0_pu": float(np.mean(vals_v)),
        "avg_L_3I0_pu": float(np.mean(vals_i)),
        "avg_ratio_V0_to_I0_pu": float(np.mean(vals_ratio)),
        "n_batches": int(len(vals_v)),
    }


# ============================================================
# Reconstruction error + collection
# ============================================================

def batch_recon_error(x_hat: torch.Tensor, x: torch.Tensor,
                      mode: Literal["mse", "mae"]) -> torch.Tensor:
    if mode == "mae":
        return (x_hat - x).abs().mean(dim=(1, 2))
    return ((x_hat - x) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def collect_frame_errors(model, loader, device, err_mode):
    model.eval()
    errs, fids, ys = [], [], []
    for x, y, fid in loader:
        x = x.to(device, non_blocking=True)
        x_hat = model(x)
        e = batch_recon_error(x_hat, x, err_mode).detach().cpu().numpy()
        errs.append(e)
        fids.append(fid.numpy().astype(np.int64))
        ys.append(y.numpy().astype(np.int64))
    if errs:
        return np.concatenate(errs), np.concatenate(fids), np.concatenate(ys)
    return np.array([]), np.array([]), np.array([])


# ============================================================
# FILE-LEVEL AGGREGATION
# ============================================================

def aggregate_file_scores(frame_scores: np.ndarray,
                          file_ids: np.ndarray,
                          n_files: int,
                          agg: Literal["max", "mean", "q"] = "q",
                          q: float = 0.95) -> np.ndarray:
    S = np.full((n_files,), np.nan, dtype=np.float64)
    for fid in range(n_files):
        s = frame_scores[file_ids == fid]
        if s.size == 0:
            S[fid] = np.nan
        elif agg == "max":
            S[fid] = float(np.max(s))
        elif agg == "mean":
            S[fid] = float(np.mean(s))
        else:
            S[fid] = float(np.quantile(s, q))
    return S


# ============================================================
# Thresholding on FILE SCORES
# ============================================================

def choose_threshold_by_percentile(scores, thr_percentile=0.995, thr_domain="log1p"):
    e = np.asarray(scores, dtype=np.float64)
    e = e[np.isfinite(e)]
    e = e[e >= 0]
    if e.size == 0:
        return float("nan"), {"status": "empty"}

    if thr_domain == "log1p":
        x = np.log1p(e)
        thr_x = float(np.quantile(x, thr_percentile))
        thr = float(np.expm1(thr_x))
        return thr, {
            "status": "ok",
            "domain": "log1p",
            "thr_percentile": thr_percentile,
            "thr_x": thr_x,
            "thr": thr,
        }

    thr = float(np.quantile(e, thr_percentile))
    return thr, {
        "status": "ok",
        "domain": "raw",
        "thr_percentile": thr_percentile,
        "thr": thr,
    }


def apply_threshold_safety(thr, scores, safety_q=0.95):
    e = np.asarray(scores, dtype=np.float64)
    e = e[np.isfinite(e)]
    e = e[e >= 0]
    if e.size == 0:
        return thr
    nfq = float(np.quantile(e, safety_q))
    return max(thr, nfq)


# ============================================================
# ROC helpers
# ============================================================

def save_roc_png(fpr, tpr, auc_val, out_png, title):
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC={auc_val:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_pr_png(recall, precision, ap_val, out_png, title):
    plt.figure()
    plt.plot(recall, precision, linewidth=2, label=f"AP={ap_val:.4f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def parse_percentile_list(s: str) -> List[float]:
    vals = []
    for item in str(s).split(','):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if not (0.0 < val < 1.0):
            raise ValueError(f"Percentile values must lie in (0, 1). Got: {val}")
        vals.append(val)
    if not vals:
        raise ValueError("No valid percentile values provided.")
    return sorted(set(vals))


def compute_binary_metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores > thr).astype(np.int64)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Positive class = HIF / fault
    hif_precision = tp / max(tp + fp, 1)
    hif_recall = tp / max(tp + fn, 1)
    hif_f1 = 2.0 * hif_precision * hif_recall / max(hif_precision + hif_recall, 1e-12)

    # Negative class = normal / non-fault (computed symmetrically)
    normal_precision = tn / max(tn + fn, 1)
    normal_recall = tn / max(tn + fp, 1)
    normal_f1 = 2.0 * normal_precision * normal_recall / max(normal_precision + normal_recall, 1e-12)

    fpr = fp / neg if neg > 0 else np.nan
    tpr = hif_recall if pos > 0 else np.nan
    tnr = normal_recall if neg > 0 else np.nan
    fnr = fn / pos if pos > 0 else np.nan
    acc = (tp + tn) / max(len(y_true), 1)

    return {
        "n_total": int(len(y_true)),
        "n_pos": pos,
        "n_neg": neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        # Backward-compatible positive-class summary columns
        "precision": float(hif_precision),
        "recall": float(hif_recall),
        "f1": float(hif_f1),
        "accuracy": float(acc),
        "tpr": float(tpr) if np.isfinite(tpr) else np.nan,
        "fpr": float(fpr) if np.isfinite(fpr) else np.nan,
        "tnr": float(tnr) if np.isfinite(tnr) else np.nan,
        "fnr": float(fnr) if np.isfinite(fnr) else np.nan,
        # Explicit per-class metrics for threshold sweep analysis
        "normal_precision": float(normal_precision),
        "normal_recall": float(normal_recall),
        "normal_f1": float(normal_f1),
        "hif_precision": float(hif_precision),
        "hif_recall": float(hif_recall),
        "hif_f1": float(hif_f1),
    }


def compute_healthy_far(scores: np.ndarray, thr: float) -> Dict[str, float]:
    e = np.asarray(scores, dtype=np.float64)
    e = e[np.isfinite(e)]
    n = int(e.size)
    if n == 0:
        return {
            "n_healthy": 0,
            "healthy_far": np.nan,
            "healthy_tnr": np.nan,
            "n_false_alarms": 0,
        }

    alarms = int(np.sum(e > thr))
    far = alarms / n
    return {
        "n_healthy": n,
        "healthy_far": float(far),
        "healthy_tnr": float(1.0 - far),
        "n_false_alarms": alarms,
    }


def bootstrap_threshold_ci(scores: np.ndarray,
                           thr_percentile: float,
                           thr_domain: str,
                           safety_q: float,
                           n_boot: int,
                           alpha: float,
                           seed: int) -> Dict[str, float]:
    e = np.asarray(scores, dtype=np.float64)
    e = e[np.isfinite(e)]
    e = e[e >= 0]
    n = int(e.size)

    if n == 0 or n_boot <= 0:
        return {
            "bootstrap_iters": int(max(n_boot, 0)),
            "bootstrap_n": n,
            "thr_boot_mean": np.nan,
            "thr_boot_std": np.nan,
            "thr_boot_ci_low": np.nan,
            "thr_boot_ci_high": np.nan,
        }

    rng = np.random.RandomState(seed)
    boots = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = e[idx]
        thr_i, _ = choose_threshold_by_percentile(sample, thr_percentile, thr_domain)
        thr_i = apply_threshold_safety(thr_i, sample, safety_q)
        boots[i] = thr_i

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    std = float(np.std(boots, ddof=1)) if n_boot > 1 else 0.0

    return {
        "bootstrap_iters": int(n_boot),
        "bootstrap_n": n,
        "thr_boot_mean": float(np.mean(boots)),
        "thr_boot_std": std,
        "thr_boot_ci_low": lo,
        "thr_boot_ci_high": hi,
    }


def build_threshold_sweep_table(threshold_fit_scores: np.ndarray,
                                healthy_val_scores: np.ndarray,
                                test_scores: np.ndarray,
                                test_y: np.ndarray,
                                percentiles: List[float],
                                thr_domain: str,
                                safety_q: float,
                                bootstrap_iters: int = 0,
                                bootstrap_alpha: float = 0.05,
                                seed: int = 42) -> pd.DataFrame:
    rows = []
    for k, p in enumerate(percentiles):
        thr, info = choose_threshold_by_percentile(threshold_fit_scores, p, thr_domain)
        thr = apply_threshold_safety(thr, threshold_fit_scores, safety_q)

        row = {
            "percentile": float(p),
            "threshold": float(thr),
            "thr_domain": str(thr_domain),
            "thr_safety_q": float(safety_q),
        }
        row.update(compute_healthy_far(healthy_val_scores, thr))
        row.update({f"test_{kk}": vv for kk, vv in compute_binary_metrics_from_scores(test_y, test_scores, thr).items()})

        if bootstrap_iters > 0:
            row.update(bootstrap_threshold_ci(
                threshold_fit_scores,
                thr_percentile=p,
                thr_domain=thr_domain,
                safety_q=safety_q,
                n_boot=bootstrap_iters,
                alpha=bootstrap_alpha,
                seed=seed + k * 9973,
            ))
        else:
            row.update({
                "bootstrap_iters": 0,
                "bootstrap_n": int(np.sum(np.isfinite(np.asarray(threshold_fit_scores, dtype=np.float64)))),
                "thr_boot_mean": np.nan,
                "thr_boot_std": np.nan,
                "thr_boot_ci_low": np.nan,
                "thr_boot_ci_high": np.nan,
            })

        if isinstance(info, dict) and "thr_x" in info:
            row["threshold_domain_value"] = float(info["thr_x"])
        else:
            row["threshold_domain_value"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_pr_curve_outputs(y_true: np.ndarray, scores: np.ndarray, out_csv: str, out_png: str, title: str) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if np.unique(y_true).size < 2:
        pd.DataFrame({"recall": [], "precision": [], "threshold": []}).to_csv(out_csv, index=False)
        return float('nan')

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap_val = float(average_precision_score(y_true, scores))

    thr_pad = np.full((len(precision),), np.nan, dtype=np.float64)
    if thresholds.size > 0:
        thr_pad[:-1] = thresholds

    pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": thr_pad,
    }).to_csv(out_csv, index=False)

    save_pr_png(recall, precision, ap_val, out_png, title)
    return ap_val



@torch.no_grad()
def evaluate_epoch_loss(model,
                        loader,
                        device,
                        rec_loss_fn,
                        t_means,
                        t_stds,
                        v_base,
                        i_base,
                        lambda_asym,
                        lambda_freq,
                        w_phys,
                        w_zero,
                        w_sym,
                        w_power,
                        w_energy):
    model.eval()
    losses = []

    comp_v = []
    comp_i = []
    comp_ratio = []

    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        x_hat = model(x)

        L_rec = rec_loss_fn(x_hat, x)
        L_asym = asymmetry_loss(x_hat, x) if lambda_asym > 0 else torch.tensor(0.0, device=device)
        L_freq = frequency_loss(x_hat, x) if lambda_freq > 0 else torch.tensor(0.0, device=device)
        L_phys = physics_loss(
            x, x_hat,
            means=t_means,
            stds=t_stds,
            v_base=v_base,
            i_base=i_base,
            w_zero=w_zero,
            w_sym=w_sym,
            w_power=w_power,
            w_energy=w_energy
        ) if w_phys > 0 else torch.tensor(0.0, device=device)

        loss = L_rec + lambda_asym * L_asym + lambda_freq * L_freq + w_phys * L_phys
        losses.append(float(loss.item()))

        if w_zero > 0:
            comps = physics_zero_seq_components(
                x, x_hat, t_means, t_stds, v_base=v_base, i_base=i_base
            )
            comp_v.append(float(comps["L_3V0_pu"].detach().item()))
            comp_i.append(float(comps["L_3I0_pu"].detach().item()))
            comp_ratio.append(float(comps["ratio_V0_to_I0_pu"].detach().item()))

    out = {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "avg_L_3V0_pu": float(np.mean(comp_v)) if comp_v else np.nan,
        "avg_L_3I0_pu": float(np.mean(comp_i)) if comp_i else np.nan,
        "avg_ratio_V0_to_I0_pu": float(np.mean(comp_ratio)) if comp_ratio else np.nan,
    }
    return out


# ============================================================
# Training loop
# ============================================================

def train_model(model,
                dl_tr,
                dl_monitor,
                device,
                scaler,
                want_input,
                v_base,
                i_base,
                epochs,
                lr,
                denoise_sigma,
                warmup_epochs,
                patience,
                min_rel_improve,
                min_abs_improve,
                lambda_asym,
                lambda_freq,
                w_phys,
                w_zero,
                w_sym,
                w_power,
                w_energy,
                clip_grad,
                out_dir,
                early_stop_on: str = "monitor"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rec_loss_fn = nn.MSELoss()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    np_means, np_stds = scaler.inv_stats_for_input(want_input)
    t_means = torch.tensor(np_means, dtype=torch.float32, device=device).view(1, 1, -1)
    t_stds = torch.tensor(np_stds, dtype=torch.float32, device=device).view(1, 1, -1)

    if early_stop_on == "monitor" and dl_monitor is None:
        print("[!] No monitor loader provided. Falling back to training loss for early stopping.", flush=True)
        early_stop_on = "train"

    best_metric = float("inf")
    bad_epochs = 0
    best_epoch = 0
    best_path = os.path.join(out_dir, "best_tcn_ae.pt")

    history = []

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        tr_comp_v = []
        tr_comp_i = []
        tr_comp_ratio = []

        for x, _, _ in dl_tr:
            x = x.to(device, non_blocking=True)
            x_in = x + denoise_sigma * torch.randn_like(x) if denoise_sigma > 0 else x

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                x_hat = model(x_in)

                L_rec = rec_loss_fn(x_hat, x)
                L_asym = asymmetry_loss(x_hat, x) if lambda_asym > 0 else 0.0
                L_freq = frequency_loss(x_hat, x) if lambda_freq > 0 else 0.0
                L_phys = physics_loss(
                    x, x_hat,
                    means=t_means,
                    stds=t_stds,
                    v_base=v_base,
                    i_base=i_base,
                    w_zero=w_zero,
                    w_sym=w_sym,
                    w_power=w_power,
                    w_energy=w_energy
                ) if w_phys > 0 else 0.0

                loss = L_rec + lambda_asym * L_asym + lambda_freq * L_freq + w_phys * L_phys

            opt.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()

            if clip_grad and clip_grad > 0:
                amp_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            amp_scaler.step(opt)
            amp_scaler.update()

            train_losses.append(float(loss.item()))

            if w_zero > 0:
                with torch.no_grad():
                    comps = physics_zero_seq_components(
                        x, x_hat, t_means, t_stds, v_base=v_base, i_base=i_base
                    )
                    tr_comp_v.append(float(comps["L_3V0_pu"].detach().item()))
                    tr_comp_i.append(float(comps["L_3I0_pu"].detach().item()))
                    tr_comp_ratio.append(float(comps["ratio_V0_to_I0_pu"].detach().item()))

        tr_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        tr_avg_v = float(np.mean(tr_comp_v)) if tr_comp_v else np.nan
        tr_avg_i = float(np.mean(tr_comp_i)) if tr_comp_i else np.nan
        tr_avg_ratio = float(np.mean(tr_comp_ratio)) if tr_comp_ratio else np.nan

        monitor_loss = None
        mon_avg_v = np.nan
        mon_avg_i = np.nan
        mon_avg_ratio = np.nan

        if dl_monitor is not None:
            mon_stats = evaluate_epoch_loss(
                model=model,
                loader=dl_monitor,
                device=device,
                rec_loss_fn=rec_loss_fn,
                t_means=t_means,
                t_stds=t_stds,
                v_base=v_base,
                i_base=i_base,
                lambda_asym=lambda_asym,
                lambda_freq=lambda_freq,
                w_phys=w_phys,
                w_zero=w_zero,
                w_sym=w_sym,
                w_power=w_power,
                w_energy=w_energy,
            )
            monitor_loss = mon_stats["loss"]
            mon_avg_v = mon_stats["avg_L_3V0_pu"]
            mon_avg_i = mon_stats["avg_L_3I0_pu"]
            mon_avg_ratio = mon_stats["avg_ratio_V0_to_I0_pu"]

        monitored_metric = monitor_loss if early_stop_on == "monitor" else tr_loss

        if np.isinf(best_metric):
            rel_gain = float("inf")
            abs_gain = float("inf")
            improved = True
        else:
            abs_gain = best_metric - monitored_metric
            rel_gain = abs_gain / max(abs(best_metric), 1e-12)
            improved = (abs_gain > min_abs_improve) and (rel_gain > min_rel_improve)

        if improved:
            best_metric = monitored_metric
            bad_epochs = 0
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
        else:
            if ep >= warmup_epochs:
                bad_epochs += 1

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "monitor_loss": monitor_loss,
            "monitored_metric": monitored_metric,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "rel_gain": None if np.isinf(rel_gain) else float(rel_gain),
            "abs_gain": None if np.isinf(abs_gain) else float(abs_gain),
            "improved": bool(improved),
            "early_stop_on": early_stop_on,
            "min_rel_improve": float(min_rel_improve),
            "min_abs_improve": float(min_abs_improve),
            "train_avg_L_3V0_pu": tr_avg_v,
            "train_avg_L_3I0_pu": tr_avg_i,
            "train_avg_ratio_V0_to_I0_pu": tr_avg_ratio,
            "monitor_avg_L_3V0_pu": mon_avg_v,
            "monitor_avg_L_3I0_pu": mon_avg_i,
            "monitor_avg_ratio_V0_to_I0_pu": mon_avg_ratio,
        })

        monitor_msg = f"{monitor_loss:.6e}" if monitor_loss is not None else "None"
        rel_msg = "inf" if np.isinf(rel_gain) else f"{100.0 * rel_gain:.4f}%"

        print(
            f"Epoch {ep:03d} | train_loss={tr_loss:.6e} | monitor_loss={monitor_msg} | "
            f"watch={monitored_metric:.6e} | best={best_metric:.6e} | "
            f"rel_gain={rel_msg} | bad_epochs={bad_epochs}/{patience} | "
            f"train_L3V0_pu={tr_avg_v:.6e} | train_L3I0_pu={tr_avg_i:.6e} | train_VI_ratio_pu={tr_avg_ratio:.6e}",
            flush=True
        )

        if ep >= warmup_epochs and bad_epochs >= patience:
            print(
                f"Early stopping triggered: monitored loss failed to improve by more than "
                f"{100.0 * min_rel_improve:.3f}% for {patience} epochs.",
                flush=True
            )
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    return best_path


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse

    ap = argparse.ArgumentParser(allow_abbrev=False)

    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--input", choices=["v", "i", "vi"], default="vi")
    ap.add_argument("--seq_len", type=int, default=8000,
                    help="Target sequence length before framing. Use <=0 to keep the full sequence.")
    ap.add_argument("--crop_mode", choices=["last", "first", "center", "random", "full"], default="last",
                    help="How to crop sequences longer than seq_len. 'full' ignores seq_len and keeps all samples.")
    ap.add_argument("--n_channels", type=int, default=3)
    ap.add_argument("--allow_channel_duplication", action="store_true",
                    help="Allow duplicating the last available channel if fewer than n_channels are found.")

    ap.add_argument("--frame_len", type=int, default=800)
    ap.add_argument("--hop", type=int, default=800)

    ap.add_argument("--hidden_ch", type=int, default=64)
    ap.add_argument("--latent_ch", type=int, default=16)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dilations", type=str, default="1,2,4,8")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_downsample", action="store_true")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--monitor_frac", type=float, default=0.15,
                    help="Fraction of healthy training files reserved for early stopping monitoring.")
    ap.add_argument("--early_stop_on", choices=["monitor", "train"], default="monitor",
                    help="Use monitor loss if available; fall back to training loss otherwise.")
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_rel_improve", type=float, default=0.002,
                    help="Minimum relative improvement required to reset patience. 0.002 = 0.2%%.")
    ap.add_argument("--min_abs_improve", type=float, default=0.0,
                    help="Optional absolute improvement floor.")

    ap.add_argument("--denoise_sigma", type=float, default=0.05)
    ap.add_argument("--err_mode", choices=["mse", "mae"], default="mae")

    ap.add_argument("--lambda_asym", type=float, default=0.0)
    ap.add_argument("--lambda_freq", type=float, default=0.0)

    ap.add_argument("--w_phys", type=float, default=0.01,
                    help="Overall multiplier for physics loss. Set 0 to disable.")
    ap.add_argument("--w_zero", type=float, default=0.0,
                    help="Weight for zero-sequence family: (3V0 + 3I0).")
    ap.add_argument("--w_sym", type=float, default=0.0,
                    help="Weight for symmetry family: (V symmetry + I symmetry).")
    ap.add_argument("--w_power", type=float, default=0.0,
                    help="Weight for power-consistency term only.")
    ap.add_argument("--w_energy", type=float, default=0.0,
                    help="Weight for energy-preservation term only.")
    ap.add_argument("--v_base", type=float, default=-1.0,
                    help="Voltage base for per-unit physics terms. Use >0 for explicit engineering base; <=0 estimates from healthy training data.")
    ap.add_argument("--i_base", type=float, default=-1.0,
                    help="Current base for per-unit physics terms. Use >0 for explicit engineering base; <=0 estimates from healthy training data.")

    ap.add_argument("--clip_grad", type=float, default=1.0)

    # file-level aggregation
    ap.add_argument("--file_agg", choices=["max", "mean", "q"], default="q")
    ap.add_argument("--file_q", type=float, default=0.95)

    # threshold on file scores
    ap.add_argument("--thr_percentile", type=float, default=0.995)
    ap.add_argument("--thr_domain", choices=["raw", "log1p"], default="log1p")
    ap.add_argument("--thr_safety_q", type=float, default=0.95)
    ap.add_argument("--sweep_percentiles", type=str, default="0.90,0.95,0.97,0.99,0.995",
                    help="Comma-separated percentile sweep for threshold sensitivity analysis.")
    ap.add_argument("--bootstrap_iters", type=int, default=0,
                    help="Optional bootstrap iterations for threshold stability CI. Set 0 to disable.")
    ap.add_argument("--bootstrap_alpha", type=float, default=0.05,
                    help="Two-sided alpha for bootstrap threshold CI.")

    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--preload", action="store_true",
                    help="Preload all transformed files into memory.")
    ap.add_argument("--max_cache_files", type=int, default=64,
                    help="LRU cache size used when preload=False.")

    args = ap.parse_args()

    # Force preloading to completely kill the multiprocessing I/O Bottleneck.
    if not args.preload:
        print("\n[!] WARNING: Forcing args.preload=True to prevent massive multiprocess I/O Bottleneck.\n")
        args.preload = True

    os.makedirs(args.out_dir, exist_ok=True)

    sweep_percentiles = parse_percentile_list(args.sweep_percentiles)
    if all(abs(p - args.thr_percentile) > 1e-12 for p in sweep_percentiles):
        sweep_percentiles = sorted(set(sweep_percentiles + [float(args.thr_percentile)]))

    seed_all(args.seed)
    device = pick_device(args.device)
    print("Device:", device, flush=True)

    # Load pairs
    train_pairs_all = build_pairs(args.train_dir)
    train_pairs = [p for p in train_pairs_all if is_nonfault_class(p.class_name)]
    if len(train_pairs) == 0:
        raise SystemExit("No non-fault files found in train_dir.")

    train_pairs_fit, monitor_pairs = split_train_monitor_pairs(
        train_pairs,
        monitor_frac=args.monitor_frac,
        seed=args.seed
    )

    test_pairs = build_pairs(args.val_dir)
    if len(test_pairs) == 0:
        raise SystemExit("No usable files found in val_dir.")

    print("TRAIN nonfault files (all healthy):", len(train_pairs), flush=True)
    print("TRAIN fit files:", len(train_pairs_fit), flush=True)
    print("MONITOR healthy files:", len(monitor_pairs), flush=True)
    print("TEST files:", len(test_pairs), flush=True)
    print(f"Sequence crop mode: {args.crop_mode} | seq_len={args.seq_len}", flush=True)

    # Fit scaler on TRAIN only
    scaler = GlobalVIScaler.fit(
        train_pairs_fit,
        args.seq_len,
        args.n_channels,
        crop_mode=args.crop_mode,
        allow_channel_duplication=args.allow_channel_duplication,
    )
    scaler.save_npz(
        os.path.join(args.out_dir, "scaler_vi_global.npz"),
        seq_len=args.seq_len, n_channels=args.n_channels, crop_mode=args.crop_mode
    )

    # Per-unit bases for physics terms
    if args.v_base > 0 and args.i_base > 0:
        v_base = float(args.v_base)
        i_base = float(args.i_base)
        pu_base_source = "explicit_cli"
    else:
        v_base, i_base = estimate_vi_bases_from_pairs(
            train_pairs_fit,
            seq_len=args.seq_len,
            n_channels=args.n_channels,
            crop_mode=args.crop_mode,
            allow_channel_duplication=args.allow_channel_duplication,
        )
        pu_base_source = "estimated_from_healthy_train"

    print(f"Per-unit bases | source={pu_base_source} | v_base={v_base:.6e} | i_base={i_base:.6e}", flush=True)

    with open(os.path.join(args.out_dir, "per_unit_bases.json"), "w") as f:
        json.dump({
            "pu_base_source": pu_base_source,
            "v_base": float(v_base),
            "i_base": float(i_base),
        }, f, indent=2)

    # Train dataset/loader (Shuffled)
    ds_tr = FrameDataset(
        train_pairs_fit,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        want_input=args.input,
        scaler=scaler,
        frame_len=args.frame_len,
        hop=args.hop,
        preload=args.preload,
        crop_mode=args.crop_mode,
        allow_channel_duplication=args.allow_channel_duplication,
        max_cache_files=args.max_cache_files,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    #  Eval dataset/loader (STRICTLY UNSHUFFLED for chronological error plotting)
    dl_eval_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    dl_monitor = None
    if len(monitor_pairs) > 0:
        ds_monitor = FrameDataset(
            monitor_pairs,
            seq_len=args.seq_len,
            n_channels=args.n_channels,
            want_input=args.input,
            scaler=scaler,
            frame_len=args.frame_len,
            hop=args.hop,
            preload=args.preload,
            crop_mode=args.crop_mode,
            allow_channel_duplication=args.allow_channel_duplication,
            max_cache_files=args.max_cache_files,
        )
        dl_monitor = DataLoader(
            ds_monitor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
            pin_memory=(device == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

    # Infer n_features safely
    x0, _, _ = ds_tr[0]
    n_features = int(x0.shape[1])

    dilations = tuple(int(x.strip()) for x in args.dilations.split(",") if x.strip())
    model = TCNAutoencoder(
        n_features=n_features,
        hidden_ch=args.hidden_ch,
        latent_ch=args.latent_ch,
        kernel_size=args.kernel_size,
        dilations=dilations,
        dropout=args.dropout,
        use_downsample=(not args.no_downsample)
    ).to(device)

    # Train
    train_model(
        model=model,
        dl_tr=dl_tr,
        dl_monitor=dl_monitor,
        device=device,
        scaler=scaler,
        want_input=args.input,
        v_base=v_base,
        i_base=i_base,
        epochs=args.epochs,
        lr=args.lr,
        denoise_sigma=args.denoise_sigma,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        min_rel_improve=args.min_rel_improve,
        min_abs_improve=args.min_abs_improve,
        lambda_asym=args.lambda_asym,
        lambda_freq=args.lambda_freq,
        w_phys=args.w_phys,
        w_zero=args.w_zero,
        w_sym=args.w_sym,
        w_power=args.w_power,
        w_energy=args.w_energy,
        clip_grad=args.clip_grad,
        out_dir=args.out_dir,
        early_stop_on=args.early_stop_on,
    )


    # Recreate stats tensors for post-training diagnostics
    np_means, np_stds = scaler.inv_stats_for_input(args.input)
    t_means = torch.tensor(np_means, dtype=torch.float32, device=device).view(1, 1, -1)
    t_stds = torch.tensor(np_stds, dtype=torch.float32, device=device).view(1, 1, -1)

    # CALIBRATION frame errors on all healthy files (fit + monitor), unshuffled
    ds_cal = FrameDataset(
        train_pairs,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        want_input=args.input,
        scaler=scaler,
        frame_len=args.frame_len,
        hop=args.hop,
        preload=args.preload,
        crop_mode=args.crop_mode,
        allow_channel_duplication=args.allow_channel_duplication,
        max_cache_files=args.max_cache_files,
    )
    dl_cal = DataLoader(
        ds_cal,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    train_frame_err, train_file_ids, _ = collect_frame_errors(model, dl_cal, device, args.err_mode)
    pd.DataFrame({
        "frame_err": train_frame_err,
        "file_id": train_file_ids
    }).to_csv(os.path.join(args.out_dir, "train_frame_errors.csv"), index=False)

    # HEALTHY calibration file scores
    train_file_scores = aggregate_file_scores(
        train_frame_err,
        train_file_ids,
        n_files=len(train_pairs),
        agg=args.file_agg,
        q=args.file_q
    )
    pd.DataFrame({
        "file_id": np.arange(len(train_pairs)),
        "file_score": train_file_scores,
        "n_frames": ds_cal.frames_per_file,
    }).to_csv(os.path.join(args.out_dir, "train_file_scores.csv"), index=False)

    healthy_val_pairs = monitor_pairs if len(monitor_pairs) > 0 else train_pairs
    healthy_val_source = "monitor_pairs" if len(monitor_pairs) > 0 else "all_healthy_train_pairs"
    ds_hval = FrameDataset(
        healthy_val_pairs,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        want_input=args.input,
        scaler=scaler,
        frame_len=args.frame_len,
        hop=args.hop,
        preload=args.preload,
        crop_mode=args.crop_mode,
        allow_channel_duplication=args.allow_channel_duplication,
        max_cache_files=args.max_cache_files,
    )
    dl_hval = DataLoader(
        ds_hval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    healthy_val_frame_err, healthy_val_file_ids, _ = collect_frame_errors(model, dl_hval, device, args.err_mode)
    healthy_val_file_scores = aggregate_file_scores(
        healthy_val_frame_err,
        healthy_val_file_ids,
        n_files=len(healthy_val_pairs),
        agg=args.file_agg,
        q=args.file_q
    )
    pd.DataFrame({
        "file_id": np.arange(len(healthy_val_pairs)),
        "file_score": healthy_val_file_scores,
        "n_frames": ds_hval.frames_per_file,
        "source": healthy_val_source,
    }).to_csv(os.path.join(args.out_dir, "healthy_validation_file_scores.csv"), index=False)
    pd.DataFrame({
        "frame_err": healthy_val_frame_err,
        "file_id": healthy_val_file_ids,
        "source": healthy_val_source,
    }).to_csv(os.path.join(args.out_dir, "healthy_validation_frame_errors.csv"), index=False)

    # Threshold on TRAIN FILE SCORES
    thr, info = choose_threshold_by_percentile(train_file_scores, args.thr_percentile, args.thr_domain)
    thr = apply_threshold_safety(thr, train_file_scores, args.thr_safety_q)

    # NEW: Threshold on TRAIN FRAME ERRORS
    thr_frame, info_frame = choose_threshold_by_percentile(train_frame_err, args.thr_percentile, args.thr_domain)
    thr_frame = apply_threshold_safety(thr_frame, train_frame_err, args.thr_safety_q)

    with open(os.path.join(args.out_dir, "threshold.txt"), "w") as f:
        f.write("level=file_and_frame\n")
        f.write(f"file_agg={args.file_agg}\n")
        f.write(f"file_q={args.file_q}\n")
        f.write("method=percentile\n")
        f.write(f"thr_percentile={args.thr_percentile}\n")
        f.write(f"thr_domain={args.thr_domain}\n")
        f.write(f"thr_safety_q={args.thr_safety_q}\n")
        f.write(f"threshold_file={thr}\n")
        f.write(f"threshold_frame={thr_frame}\n")
        f.write(f"crop_mode={args.crop_mode}\n")
        f.write(f"healthy_validation_source={healthy_val_source}\n")
        f.write(f"healthy_validation_file_far={compute_healthy_far(healthy_val_file_scores, thr)['healthy_far']}\n")
        f.write(f"healthy_validation_frame_far={compute_healthy_far(healthy_val_frame_err, thr_frame)['healthy_far']}\n")
        f.write("frame_labels=weak_labels_inherited_from_file_labels\n")
        f.write(f"info_file={info}\n")
        f.write(f"info_frame={info_frame}\n")

    print("Chosen FILE threshold (after safety rule):", thr, flush=True)
    print("Chosen FRAME threshold (after safety rule):", thr_frame, flush=True)

    # TEST dataset/loader
    ds_test = FrameDataset(
        test_pairs,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        want_input=args.input,
        scaler=scaler,
        frame_len=args.frame_len,
        hop=args.hop,
        preload=args.preload,
        crop_mode=args.crop_mode,
        allow_channel_duplication=args.allow_channel_duplication,
        max_cache_files=args.max_cache_files,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ============================================================
    # Zero-sequence diagnostics on healthy/test loaders
    # ============================================================

    zero_seq_diagnostics = {}

    if args.w_zero > 0:
        zero_seq_diagnostics["healthy_calibration"] = evaluate_zero_seq_components(
            model=model,
            loader=dl_cal,
            device=device,
            means=t_means,
            stds=t_stds,
            v_base=v_base,
            i_base=i_base,
        )

        zero_seq_diagnostics["healthy_validation"] = evaluate_zero_seq_components(
            model=model,
            loader=dl_hval,
            device=device,
            means=t_means,
            stds=t_stds,
            v_base=v_base,
            i_base=i_base,
        )

        zero_seq_diagnostics["test_all"] = evaluate_zero_seq_components(
            model=model,
            loader=dl_test,
            device=device,
            means=t_means,
            stds=t_stds,
            v_base=v_base,
            i_base=i_base,
        )
    else:
        zero_seq_diagnostics["healthy_calibration"] = {
            "avg_L_3V0_pu": np.nan,
            "avg_L_3I0_pu": np.nan,
            "avg_ratio_V0_to_I0_pu": np.nan,
            "n_batches": 0,
        }
        zero_seq_diagnostics["healthy_validation"] = {
            "avg_L_3V0_pu": np.nan,
            "avg_L_3I0_pu": np.nan,
            "avg_ratio_V0_to_I0_pu": np.nan,
            "n_batches": 0,
        }
        zero_seq_diagnostics["test_all"] = {
            "avg_L_3V0_pu": np.nan,
            "avg_L_3I0_pu": np.nan,
            "avg_ratio_V0_to_I0_pu": np.nan,
            "n_batches": 0,
        }

    pd.DataFrame([
        {"split": k, **v} for k, v in zero_seq_diagnostics.items()
    ]).to_csv(os.path.join(args.out_dir, "zero_seq_component_summary.csv"), index=False)

    with open(os.path.join(args.out_dir, "zero_seq_component_summary.json"), "w") as f:
        json.dump(zero_seq_diagnostics, f, indent=2)

    # TEST frame errors
    test_frame_err, test_file_ids, test_y_frame = collect_frame_errors(model, dl_test, device, args.err_mode)
    pd.DataFrame({
        "frame_err": test_frame_err,
        "file_id": test_file_ids,
        "y_true_binary_weak": test_y_frame,
    }).to_csv(os.path.join(args.out_dir, "test_frame_errors.csv"), index=False)

    # Weak-label frame ROC (diagnostic only)
    fpr, tpr, _ = roc_curve(test_y_frame, test_frame_err)
    auc_frame = auc(fpr, tpr)
    pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(
        os.path.join(args.out_dir, "roc_frame_weak.csv"),
        index=False
    )
    save_roc_png(
        fpr, tpr, auc_frame,
        os.path.join(args.out_dir, "roc_frame_weak.png"),
        "Weak-label frame ROC (diagnostic only, labels inherited from file status)"
    )

    # TEST file scores
    test_file_scores = aggregate_file_scores(
        test_frame_err,
        test_file_ids,
        n_files=len(test_pairs),
        agg=args.file_agg,
        q=args.file_q
    )
    y_true_by_file = ds_test.y_by_file

    pd.DataFrame({
        "file_id": np.arange(len(test_pairs)),
        "file_score": test_file_scores,
        "y_true": y_true_by_file,
        "n_frames": ds_test.frames_per_file,
    }).to_csv(os.path.join(args.out_dir, "test_file_scores.csv"), index=False)

    # File ROC from file scores (primary metric)
    fpr_f, tpr_f, _ = roc_curve(y_true_by_file, test_file_scores)
    auc_file = auc(fpr_f, tpr_f)
    pd.DataFrame({"FPR": fpr_f, "TPR": tpr_f}).to_csv(
        os.path.join(args.out_dir, "roc_file.csv"),
        index=False
    )
    save_roc_png(
        fpr_f, tpr_f, auc_file,
        os.path.join(args.out_dir, "roc_file.png"),
        "File-level ROC (primary metric, file score)"
    )
    print("Weak-label frame AUC (diagnostic only):", auc_frame, flush=True)
    print("File-level AUC (primary metric):", auc_file, flush=True)

    # PR curves and AP
    ap_frame = save_pr_curve_outputs(
        test_y_frame,
        test_frame_err,
        os.path.join(args.out_dir, "pr_frame_weak.csv"),
        os.path.join(args.out_dir, "pr_frame_weak.png"),
        "Weak-label frame PR curve (diagnostic only)"
    )
    ap_file = save_pr_curve_outputs(
        y_true_by_file,
        test_file_scores,
        os.path.join(args.out_dir, "pr_file.csv"),
        os.path.join(args.out_dir, "pr_file.png"),
        "File-level PR curve (primary metric)"
    )
    print("Weak-label frame AP (diagnostic only):", ap_frame, flush=True)
    print("File-level AP (primary metric):", ap_file, flush=True)

    # Threshold sensitivity sweep
    df_sweep_file = build_threshold_sweep_table(
        threshold_fit_scores=train_file_scores,
        healthy_val_scores=healthy_val_file_scores,
        test_scores=test_file_scores,
        test_y=y_true_by_file,
        percentiles=sweep_percentiles,
        thr_domain=args.thr_domain,
        safety_q=args.thr_safety_q,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_alpha=args.bootstrap_alpha,
        seed=args.seed,
    )
    df_sweep_file["selected_percentile"] = np.isclose(df_sweep_file["percentile"].values, args.thr_percentile)
    df_sweep_file.to_csv(os.path.join(args.out_dir, "threshold_sweep_file.csv"), index=False)

    df_sweep_frame = build_threshold_sweep_table(
        threshold_fit_scores=train_frame_err,
        healthy_val_scores=healthy_val_frame_err,
        test_scores=test_frame_err,
        test_y=test_y_frame,
        percentiles=sweep_percentiles,
        thr_domain=args.thr_domain,
        safety_q=args.thr_safety_q,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_alpha=args.bootstrap_alpha,
        seed=args.seed + 123,
    )
    df_sweep_frame["selected_percentile"] = np.isclose(df_sweep_frame["percentile"].values, args.thr_percentile)
    df_sweep_frame.to_csv(os.path.join(args.out_dir, "threshold_sweep_frame_weak.csv"), index=False)

    healthy_file_alarm = compute_healthy_far(healthy_val_file_scores, thr)
    healthy_frame_alarm = compute_healthy_far(healthy_val_frame_err, thr_frame)

    with open(os.path.join(args.out_dir, "threshold_analysis_notes.txt"), "w") as f:
        f.write("Threshold sensitivity analysis uses thresholds fitted on healthy calibration scores.\n")
        f.write(f"Healthy-only FAR evaluation source: {healthy_val_source}\n")
        f.write("On a healthy-only set, TPR is undefined because no positive samples exist.\n")
        f.write("Accordingly, the sweep tables report healthy FAR/TNR on healthy validation and TPR/FPR on test data.\n")
        f.write("The sweep CSVs also include explicit per-class metrics: normal_precision/recall/F1 and hif_precision/recall/F1.\n")
        f.write(f"selected_file_threshold={thr}\n")
        f.write(f"selected_frame_threshold={thr_frame}\n")
        f.write(f"selected_file_healthy_far={healthy_file_alarm['healthy_far']}\n")
        f.write(f"selected_frame_healthy_far={healthy_frame_alarm['healthy_far']}\n")
        f.write(f"file_ap={ap_file}\n")
        f.write(f"frame_ap_weak={ap_frame}\n")
        f.write("See threshold_sweep_file.csv and threshold_sweep_frame_weak.csv for per-threshold per-class metrics.\n")

    if args.bootstrap_iters > 0:
        boot_file = bootstrap_threshold_ci(
            train_file_scores,
            thr_percentile=args.thr_percentile,
            thr_domain=args.thr_domain,
            safety_q=args.thr_safety_q,
            n_boot=args.bootstrap_iters,
            alpha=args.bootstrap_alpha,
            seed=args.seed,
        )
        boot_frame = bootstrap_threshold_ci(
            train_frame_err,
            thr_percentile=args.thr_percentile,
            thr_domain=args.thr_domain,
            safety_q=args.thr_safety_q,
            n_boot=args.bootstrap_iters,
            alpha=args.bootstrap_alpha,
            seed=args.seed + 12345,
        )
        with open(os.path.join(args.out_dir, "threshold_bootstrap_summary.json"), "w") as f:
            json.dump({
                "file": {"selected_threshold": thr, **boot_file},
                "frame_weak": {"selected_threshold": thr_frame, **boot_frame},
            }, f, indent=2)

    # Final evaluation at chosen FILE threshold
    y_pred_file = (test_file_scores > thr).astype(np.int64)

    cm = confusion_matrix(y_true_by_file, y_pred_file, labels=[0, 1])
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(
        os.path.join(args.out_dir, "confusion_test_binary.csv")
    )

    rep = classification_report(y_true_by_file, y_pred_file, digits=4)
    with open(os.path.join(args.out_dir, "report_test.txt"), "w") as f:
        f.write("PRIMARY EVALUATION: file-level classification\n")
        f.write("NOTE: frame-level labels are weak labels inherited from file labels.\n\n")
        f.write(rep)

    with open(os.path.join(args.out_dir, "evaluation_notes.txt"), "w") as f:
        f.write("This experiment uses weak frame labels inherited from file labels.\n")
        f.write("Therefore, roc_frame_weak.* is a diagnostic proxy metric, not a true frame-localization benchmark.\n")
        f.write("The primary metrics for model selection and reporting are file-level ROC, file-level AUC,\n")
        f.write("and the file-level classification report at the selected file threshold.\n")
        f.write(f"frame_auc_weak={auc_frame}\n")
        f.write(f"file_auc_primary={auc_file}\n")
        f.write(f"frame_ap_weak={ap_frame}\n")
        f.write(f"file_ap_primary={ap_file}\n")
        f.write(f"healthy_validation_source={healthy_val_source}\n")
        f.write(f"crop_mode={args.crop_mode}\n")
        f.write(f"seq_len={args.seq_len}\n")
        f.write(f"preload={args.preload}\n")
        f.write(f"max_cache_files={args.max_cache_files}\n")
        f.write(f"pu_base_source={pu_base_source}\n")
        f.write(f"v_base={v_base}\n")
        f.write(f"i_base={i_base}\n")

    config_dump = vars(args).copy()
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

    print("\nTEST classification report (file-level, primary metric):\n", rep, flush=True)

    # --- ADDED: Frame-level Evaluation ---
    y_pred_frame = (test_frame_err > thr_frame).astype(np.int64)

    rep_frame = classification_report(test_y_frame, y_pred_frame, digits=4)

    with open(os.path.join(args.out_dir, "report_test_frame_weak.txt"), "w") as f:
        f.write("DIAGNOSTIC EVALUATION: Frame-level classification (Weak Labels)\n")
        f.write(f"Frame Threshold used: {thr_frame}\n\n")
        f.write(rep_frame)

    print("\nFRAME-level classification report (Weak Labels):\n", rep_frame, flush=True)
    print("Saved all outputs to:", args.out_dir, flush=True)
if __name__ == "__main__":
    main()