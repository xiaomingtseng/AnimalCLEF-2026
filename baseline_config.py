"""Shared baseline configuration for official-score reproduction.

Keep similarity->distance conversion and default DBSCAN eps values in one place
so all entry scripts follow identical clustering semantics.
"""

from __future__ import annotations

import numpy as np

OFFICIAL_EPS = {
    "LynxID2025": 0.3,
    "SalamanderID2025": 0.2,
    "SeaTurtleID2022": 0.4,
    "TexasHornedLizards": 0.24,
}


def similarity_to_distance(similarity: np.ndarray) -> np.ndarray:
    """Convert similarity matrix to distance matrix using normalization.
    
    This matches the official AnimalCLEF2026 starter notebook formula.
    """
    max_sim = np.max(similarity)
    distance = (max_sim - np.maximum(similarity, 0.0)) / max_sim
    return distance