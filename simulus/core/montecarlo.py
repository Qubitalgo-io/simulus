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
    convergence_error: float = 0.0

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

    leaf_ids = set(n.node_id for n in leaves)

    outcome_counts: dict[str, int] = {leaf.node_id: 0 for leaf in leaves}
    outcome_labels: dict[str, str] = {}
    outcome_sentiments: dict[str, Sentiment] = {}

    for leaf in leaves:
        outcome_labels[leaf.node_id] = leaf.label
        outcome_sentiments[leaf.node_id] = leaf.sentiment

    rng = np.random.default_rng(seed_mgr.base_seed)

    # stochastic tree walk: at each internal node, sample a child
    # proportional to the edge probability, with small per-walk noise.
    # this means each simulation can follow a different path through
    # the tree, rather than re-sampling from a pre-computed leaf
    # distribution.
    noise_scale = 0.02

    for _ in range(n_simulations):
        current_id = cg.root_id
        while current_id not in leaf_ids:
            children = cg.get_children(current_id)
            if not children:
                break

            edge_probs = np.array([
                cg.get_edge(current_id, c.node_id).probability
                for c in children
            ], dtype=np.float64)

            # per-walk noise: each simulation is a slightly different
            # realization of the transition probabilities
            noise = rng.normal(0.0, noise_scale, size=len(edge_probs))
            perturbed = np.maximum(edge_probs + noise, 0.001)
            perturbed /= perturbed.sum()

            chosen_idx = rng.choice(len(children), p=perturbed)
            current_id = children[chosen_idx].node_id

        if current_id in outcome_counts:
            outcome_counts[current_id] += 1

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

    # convergence diagnostic: maximum absolute difference between the
    # MC-estimated leaf probabilities and the tree-computed probabilities.
    # with sufficient samples this should be small (< 0.02 for N=10000).
    leaf_prob_map = {}
    total_tree_prob = sum(l.probability for l in leaves)
    for leaf in leaves:
        tree_p = leaf.probability / total_tree_prob if total_tree_prob > 0 else 0.0
        mc_p = outcome_counts[leaf.node_id] / n_simulations if n_simulations > 0 else 0.0
        leaf_prob_map[leaf.node_id] = abs(tree_p - mc_p)
    convergence_error = max(leaf_prob_map.values()) if leaf_prob_map else 0.0

    return SimulationResult(
        n_simulations=n_simulations,
        outcome_counts=outcome_counts,
        outcome_labels=outcome_labels,
        outcome_sentiments=outcome_sentiments,
        sentiment_distribution=sentiment_distribution,
        mean_sentiment_score=mean_sentiment,
        convergence_error=convergence_error,
    )
