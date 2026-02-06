from __future__ import annotations

import hashlib
import numpy as np


class SeedManager:
    """All randomness in Simulus flows through this object.
    Given the same seed and the same sequence of calls, the output
    is byte-identical. Never use random.random() directly."""

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = int.from_bytes(hashlib.sha256(b"simulus-default").digest()[:4], "big")
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    @property
    def base_seed(self) -> int:
        return self._base_seed

    def random(self) -> float:
        self._call_count += 1
        return float(self._rng.random())

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        self._call_count += 1
        return float(self._rng.uniform(low, high))

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        self._call_count += 1
        return float(self._rng.normal(mean, std))

    def choice(self, items: list, p: list[float] | None = None):
        self._call_count += 1
        idx = self._rng.choice(len(items), p=p)
        return items[idx]

    def integers(self, low: int, high: int) -> int:
        self._call_count += 1
        return int(self._rng.integers(low, high))

    def fork(self, branch_id: str) -> SeedManager:
        """Create a child seed manager for a specific branch.
        The child seed is deterministically derived from the parent
        seed and the branch identifier, so parallel branches are
        independent but reproducible."""
        combined = f"{self._base_seed}:{branch_id}"
        child_seed = int(hashlib.sha256(combined.encode()).digest()[:4].hex(), 16)
        return SeedManager(seed=child_seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._base_seed)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count
