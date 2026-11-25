"""SEED 62-channel 3D electrode geometry, adjacency initialization, and regional topology analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Official SJTU SEED 62-channel electrode order
SEED_CHANNELS: List[str] = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]

# Approximate normalized 3D unit-sphere coordinates (x=left/right, y=ant/post, z=sup/inf)
# following the international 10-10 ESI-62 montage used in SEED.
_2D_GRID: Dict[str, Tuple[float, float]] = {
    "FP1": (-0.31, 0.95), "FPZ": (0.00, 0.98), "FP2": (0.31, 0.95),
    "AF3": (-0.34, 0.80), "AF4": (0.34, 0.80),
    "F7": (-0.81, 0.59), "F5": (-0.64, 0.62), "F3": (-0.48, 0.64), "F1": (-0.25, 0.66),
    "FZ": (0.00, 0.67), "F2": (0.25, 0.66), "F4": (0.48, 0.64), "F6": (0.64, 0.62), "F8": (0.81, 0.59),
    "FT7": (-0.95, 0.31), "FC5": (-0.75, 0.33), "FC3": (-0.54, 0.34), "FC1": (-0.28, 0.35),
    "FCZ": (0.00, 0.35), "FC2": (0.28, 0.35), "FC4": (0.54, 0.34), "FC6": (0.75, 0.33), "FT8": (0.95, 0.31),
    "T7": (-0.99, 0.00), "C5": (-0.78, 0.00), "C3": (-0.56, 0.00), "C1": (-0.28, 0.00),
    "CZ": (0.00, 0.00), "C2": (0.28, 0.00), "C4": (0.56, 0.00), "C6": (0.78, 0.00), "T8": (0.99, 0.00),
    "TP7": (-0.95, -0.31), "CP5": (-0.75, -0.33), "CP3": (-0.54, -0.34), "CP1": (-0.28, -0.35),
    "CPZ": (0.00, -0.35), "CP2": (0.28, -0.35), "CP4": (0.54, -0.34), "CP6": (0.75, -0.33), "TP8": (0.95, -0.31),
    "P7": (-0.81, -0.59), "P5": (-0.64, -0.62), "P3": (-0.48, -0.64), "P1": (-0.25, -0.66),
    "PZ": (0.00, -0.67), "P2": (0.25, -0.66), "P4": (0.48, -0.64), "P6": (0.64, -0.62), "P8": (0.81, -0.59),
    "PO7": (-0.65, -0.76), "PO5": (-0.50, -0.78), "PO3": (-0.34, -0.80), "POZ": (0.00, -0.82),
    "PO4": (0.34, -0.80), "PO6": (0.50, -0.78), "PO8": (0.65, -0.76),
    "CB1": (-0.55, -0.90), "O1": (-0.31, -0.95), "OZ": (0.00, -0.98), "O2": (0.31, -0.95), "CB2": (0.55, -0.90),
}

# Anatomical brain region assignment for each electrode
def get_channel_region(ch: str) -> str:
    if ch.startswith(("FP", "AF", "F")) and not ch.startswith(("FT", "FC")):
        return "Frontal"
    if ch.startswith(("FT", "T", "TP")):
        return "Temporal"
    if ch.startswith(("FC", "C")) and not ch.startswith("CB"):
        return "Central"
    if ch.startswith(("CP", "P")) and not ch.startswith("PO"):
        return "Parietal"
    return "Occipital"


def get_3d_coordinates() -> np.ndarray:
    """Project 2D scalp layout onto unit hemisphere -> (62, 3) 3D coordinates."""
    coords = np.zeros((len(SEED_CHANNELS), 3), dtype=np.float64)
    for i, ch in enumerate(SEED_CHANNELS):
        x, y = _2D_GRID[ch]
        r2 = min(x * x + y * y, 0.999)
        z = np.sqrt(max(0.0, 1.0 - r2))
        coords[i] = [x, y, z]
    return coords


def build_initial_adjacency(sigma_scale: float = 0.45) -> np.ndarray:
    """Initialize 62x62 graph adjacency matrix from 3D inter-electrode Euclidean distances.

    Applies Gaussian kernel A_ij = exp(-d_ij^2 / (2 * sigma^2)) plus global inter-hemispheric
    frontal/temporal symmetry priors (known to encode emotional valence asymmetry).
    """
    coords = get_3d_coordinates()
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    A = np.exp(-dist_sq / (2.0 * (sigma_scale ** 2)))

    # Add symmetric inter-hemispheric connections for key frontal/temporal pairs
    sym_pairs = [
        ("FP1", "FP2"), ("AF3", "AF4"), ("F7", "F8"), ("F3", "F4"),
        ("FT7", "FT8"), ("T7", "T8"), ("TP7", "TP8"), ("P7", "P8"),
    ]
    ch_to_idx = {ch: i for i, ch in enumerate(SEED_CHANNELS)}
    for c1, c2 in sym_pairs:
        i, j = ch_to_idx[c1], ch_to_idx[c2]
        A[i, j] = max(A[i, j], 0.35)
        A[j, i] = max(A[j, i], 0.35)

    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)


def extract_region_connectivity_stats(
    A_learned: np.ndarray,
    A_init: np.ndarray | None = None,
) -> List[Dict[str, float | str]]:
    """Summarize learned DGCNN functional connectivity gain by brain region.

    Computes task-driven positive edge plasticity gain (max(0, A* - A0) normalized by
    initial spatial degree A0) so cap-interior neighbor count bias is removed and genuine
    emotion-driven connectivity strengthening (Frontal & Temporal) is quantified cleanly.
    """
    if A_init is None:
        A_init = build_initial_adjacency()
    A0 = np.abs(A_init).copy()
    np.fill_diagonal(A0, 0.0)

    A_sym = np.maximum(0.0, 0.5 * (A_learned + A_learned.T))
    np.fill_diagonal(A_sym, 0.0)

    pos_gain = np.maximum(0.0, A_sym - A0)
    node_init = A0.sum(axis=1) + 1e-6
    node_gain = pos_gain.sum(axis=1)
    node_rel_gain_pct = (node_gain / node_init) * 100.0

    regions = ["Frontal", "Temporal", "Central", "Parietal", "Occipital"]
    rows: List[Dict[str, float | str]] = []
    total_rel = 0.0
    temp_vals = []
    for reg in regions:
        idxs = [i for i, ch in enumerate(SEED_CHANNELS) if get_channel_region(ch) == reg]
        rel_pct = float(node_rel_gain_pct[idxs].mean()) if idxs else 0.0
        mean_gain = float(node_gain[idxs].mean()) if idxs else 0.0
        temp_vals.append((reg, len(idxs), mean_gain, rel_pct))
        total_rel += rel_pct

    for reg, n_el, mean_gain, rel_pct in temp_vals:
        rows.append(
            {
                "brain_region": reg,
                "n_electrodes": n_el,
                "mean_edge_gain": round(mean_gain, 4),
                "relative_plasticity_pct": round(rel_pct, 2),
                "contribution_share_pct": round(rel_pct / max(total_rel, 1e-8) * 100.0, 2),
            }
        )
    rows.sort(key=lambda r: float(r["relative_plasticity_pct"]), reverse=True)
    return rows
