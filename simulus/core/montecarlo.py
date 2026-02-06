from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulus.core.causal_graph import CausalGraph, CausalNode, Sentiment
from simulus.seed import SeedManager


@dataclass
class SimulationResult:
    n_simulations: int
    outcome_counts: dict[str, int] = field(default_factory=dict)
    outcome_labels: dict[str, str] = field(default_factory=dict)
    outcome_sentiments: dict[str, Sentiment] = field(default_factory=dict)
    sentiment_distribution: dict[str, float] = field(default_factory=dict)
    mean_sentiment_score: float = 0.0

    @property
    def most_likely_outcome(self) -> str:
        if not self.outcome_counts:
            return "unknown"
        best_id = max(self.outcome_counts, key=self.outcome_counts.get)
        return self.outcome_labels.get(best_id, best_id)

    @property
    def least_likely_outcome(self) -> str:
        if not self.outcome_counts:
            return "unknown"
        worst_id = min(self.outcome_counts, key=self.outcome_counts.get)
        return self.outcome_labels.get(worst_id, worst_id)

    def outcome_probability(self, outcome_id: str) -> float:
        if self.n_simulations == 0:
            return 0.0
        return self.outcome_counts.get(outcome_id, 0) / self.n_simulations

    def top_outcomes(self, n: int = 5) -> list[tuple[str, float]]:
        sorted_outcomes = sorted(self.outcome_counts.items(),
                                 key=lambda x: x[1], reverse=True)
        return [(self.outcome_labels.get(oid, oid), count / self.n_simulations)
                for oid, count in sorted_outcomes[:n]]


def run_monte_carlo(cg: CausalGraph, seed_mgr: SeedManager,
                    n_simulations: int = 10000) -> SimulationResult:
    leaves = cg.get_leaves()
    if not leaves:
        return SimulationResult(n_simulations=0)

    leaf_ids = [leaf.node_id for leaf in leaves]
    leaf_probs = np.array([leaf.probability for leaf in leaves])

    total = leaf_probs.sum()
    if total <= 0:
        leaf_probs = np.ones(len(leaves)) / len(leaves)
    else:
        leaf_probs = leaf_probs / total

    outcome_counts: dict[str, int] = {lid: 0 for lid in leaf_ids}
    outcome_labels: dict[str, str] = {}
    outcome_sentiments: dict[str, Sentiment] = {}

    for leaf in leaves:
        outcome_labels[leaf.node_id] = leaf.label
        outcome_sentiments[leaf.node_id] = leaf.sentiment

    rng = np.random.default_rng(seed_mgr.base_seed)
    samples = rng.choice(len(leaf_ids), size=n_simulations, p=leaf_probs)

    for idx in samples:
        outcome_counts[leaf_ids[idx]] += 1

    sentiment_values = {
        Sentiment.POSITIVE: 1.0,
        Sentiment.NEGATIVE: -1.0,
        Sentiment.NEUTRAL: 0.0,
    }

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_sentiment = 0.0

    for leaf_id, count in outcome_counts.items():
        sentiment = outcome_sentiments[leaf_id]
        sentiment_counts[sentiment.value] += count
        total_sentiment += sentiment_values[sentiment] * count

    sentiment_distribution = {
        k: v / n_simulations for k, v in sentiment_counts.items()
    }

    mean_sentiment = total_sentiment / n_simulations if n_simulations > 0 else 0.0

    return SimulationResult(
        n_simulations=n_simulations,
        outcome_counts=outcome_counts,
        outcome_labels=outcome_labels,
        outcome_sentiments=outcome_sentiments,
        sentiment_distribution=sentiment_distribution,
        mean_sentiment_score=mean_sentiment,
    )
