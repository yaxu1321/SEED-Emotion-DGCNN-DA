"""SJTU SEED dataset loader for 62-channel 5-band DE & PSD graph features.

Reads the official SEED `ExtractedFeatures/*.mat` + `label.mat` files directly.
Each subject has 3 sessions (`{subj}_{date}.mat`), each containing 15 movie-clip trials
with 5-band (Delta, Theta, Alpha, Beta, Gamma) Differential Entropy (DE) and Power
Spectral Density (PSD) features of shape (62, T_trial, 5).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger("seed_dgcnn")

# Official SEED 15-trial labels (-1=Negative, 0=Neutral, 1=Positive) as fallback if label.mat absent
DEFAULT_SEED_LABELS = np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1], dtype=np.int64)


def _discover_subject_files(data_dir: Path) -> Dict[int, List[Path]]:
    """Group SEED .mat files by subject ID (1..15) sorted chronologically by session."""
    by_subj: Dict[int, List[Path]] = {}
    for p in sorted(data_dir.glob("*.mat")):
        m = re.match(r"^(\d+)_(\d+)\.mat$", p.name)
        if not m:
            continue
        subj = int(m.group(1))
        by_subj.setdefault(subj, []).append(p)
    for subj in by_subj:
        by_subj[subj].sort(key=lambda x: x.name)
    return by_subj


def _load_trial_labels(data_dir: Path) -> np.ndarray:
    label_file = data_dir / "label.mat"
    if label_file.exists():
        mat = loadmat(str(label_file))
        raw = np.squeeze(mat["label"]).astype(np.int64)
        # Map {-1, 0, 1} -> {0, 1, 2}
        return raw + 1
    return DEFAULT_SEED_LABELS + 1


def _load_single_session_mat(
    mat_path: Path,
    trial_labels: np.ndarray,
    feature_type: str = "de_psd",
    smoothing: str = "LDS",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load all 15 trials from one SEED ExtractedFeatures .mat file.

    Returns:
        trial_features: list of 15 arrays, each of shape (T_trial, 62, F) where F=10 (de_psd) or 5 (de).
        trial_labels: shape (15,) with values in {0, 1, 2}.
    """
    mat = loadmat(str(mat_path))
    de_prefix = f"de_{smoothing}"
    psd_prefix = f"psd_{smoothing}"

    trials: List[np.ndarray] = []
    for t_idx in range(1, 16):
        de_key = f"{de_prefix}{t_idx}"
        if de_key not in mat:
            raise KeyError(f"Expected key '{de_key}' not found in {mat_path.name}")
        de_arr = mat[de_key]  # (62, T_trial, 5)
        # Transpose to (T_trial, 62, 5)
        de_arr = np.transpose(de_arr, (1, 0, 2)).astype(np.float32)

        if feature_type == "de_psd":
            psd_key = f"{psd_prefix}{t_idx}"
            psd_arr = np.transpose(mat[psd_key], (1, 0, 2)).astype(np.float32)
            # Log-compress PSD so its dynamic range matches DE cleanly
            psd_arr = np.log1p(np.maximum(psd_arr, 0.0))
            feat = np.concatenate([de_arr, psd_arr], axis=-1)  # (T_trial, 62, 10)
        else:
            feat = de_arr  # (T_trial, 62, 5)
        trials.append(feat)

    return trials, trial_labels


def load_seed_dataset(
    subjects: Sequence[int],
    cfg: Dict,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Load SEED ExtractedFeatures for requested subjects.

    Returns per-subject dict containing:
      - 'X_all': (N_total, 62, F) channel-standardized node features
      - 'y_all': (N_total,) labels in {0, 1, 2}
      - 'X_within_train', 'y_within_train': first 9 trials (official within-subject train)
      - 'X_within_test', 'y_within_test': last 6 trials (official within-subject test)
    """
    data_dir = Path(cfg["project"]["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"SEED ExtractedFeatures directory not found at {data_dir}. "
            "Please verify SEED/ExtractedFeatures is uploaded to data/SEED/ExtractedFeatures."
        )

    dcfg = cfg["dataset"]
    by_subj = _discover_subject_files(data_dir)
    trial_labels = _load_trial_labels(data_dir)
    sess_idx = dcfg.get("session_index", 0)
    n_train_trials = int(dcfg.get("within_train_trials", 9))

    result: Dict[int, Dict[str, np.ndarray]] = {}
    for subj in subjects:
        subj = int(subj)
        if subj not in by_subj:
            raise FileNotFoundError(f"No SEED .mat files found for Subject {subj} in {data_dir}")

        files = by_subj[subj]
        selected_files = [files[sess_idx]] if sess_idx is not None else files

        tr_X_list, tr_y_list = [], []
        te_X_list, te_y_list = [], []

        for fpath in selected_files:
            trials, labels = _load_single_session_mat(
                fpath,
                trial_labels=trial_labels,
                feature_type=str(dcfg["feature_type"]),
                smoothing=str(dcfg["smoothing"]),
            )
            for idx, (X_t, lbl) in enumerate(zip(trials, labels)):
                y_t = np.full(X_t.shape[0], int(lbl), dtype=np.int64)
                if idx < n_train_trials:
                    tr_X_list.append(X_t)
                    tr_y_list.append(y_t)
                else:
                    te_X_list.append(X_t)
                    te_y_list.append(y_t)

        X_tr = np.concatenate(tr_X_list, axis=0)
        y_tr = np.concatenate(tr_y_list, axis=0)
        X_te = np.concatenate(te_X_list, axis=0)
        y_te = np.concatenate(te_y_list, axis=0)

        # Within-subject block-adaptive unsupervised normalization (eliminates 50-min slow electrode impedance drift; uses zero labels)
        mean_tr = X_tr.mean(axis=0, keepdims=True)
        std_tr = X_tr.std(axis=0, keepdims=True) + 1e-6
        mean_te = X_te.mean(axis=0, keepdims=True)
        std_te = X_te.std(axis=0, keepdims=True) + 1e-6
        X_tr_norm = (X_tr - mean_tr) / std_tr
        X_te_norm = (X_te - mean_te) / std_te

        # Full-subject unsupervised z-score for cross-subject LOSO (uses zero labels)
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all = np.concatenate([y_tr, y_te], axis=0)
        mean_a = X_all.mean(axis=0, keepdims=True)
        std_a = X_all.std(axis=0, keepdims=True) + 1e-6
        X_all_norm = (X_all - mean_a) / std_a

        result[subj] = {
            "X_all": X_all_norm.astype(np.float32),
            "y_all": y_all,
            "X_within_train": X_tr_norm.astype(np.float32),
            "y_within_train": y_tr,
            "X_within_test": X_te_norm.astype(np.float32),
            "y_within_test": y_te,
        }
        logger.info("[SEED] Subject %02d loaded: total=%s, within_train=%d, within_test=%d",
                    subj, X_all_norm.shape, len(y_tr), len(y_te))

    return result
