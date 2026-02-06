from __future__ import annotations

import math

from simulus.core.causal_graph import CausalGraph, CausalNode, NodeType
from simulus.core.parser import SituationContext
from simulus.seed import SeedManager


BASE_LYAPUNOV_EXPONENT = 0.47

# domain-specific volatility floors -- some domains are inherently more chaotic
DOMAIN_VOLATILITY: dict[str, float] = {
    "finance": 0.65,
    "career": 0.50,
    "relationship": 0.55,
    "health": 0.40,
    "education": 0.35,
    "travel": 0.45,
    "general": 0.47,
}


def compute_adaptive_exponent(context: SituationContext | None = None) -> float:
    if context is None:
        return BASE_LYAPUNOV_EXPONENT

    domain_base = DOMAIN_VOLATILITY.get(context.domain, BASE_LYAPUNOV_EXPONENT)

    # compound_volatility factors in time pressure, irreversibility, conflicts
    volatility_boost = context.compound_volatility * 0.3

    # stake severity amplifies chaos -- higher stakes, less predictable
    severity_boost = context.stake_severity * 0.15

    # actor count: more actors means more interaction effects
    actor_count = len(context.actor_profiles)
    interaction_boost = min(actor_count * 0.04, 0.16)

    exponent = domain_base + volatility_boost + severity_boost + interaction_boost

    return max(0.2, min(1.2, exponent))


def lyapunov_multiplier(depth: int, exponent: float = BASE_LYAPUNOV_EXPONENT) -> float:
    return math.exp(exponent * depth)


def apply_perturbation(probability: float, perturbation: float,
                       depth: int,
                       context: SituationContext | None = None) -> float:
    exponent = compute_adaptive_exponent(context)
    multiplier = lyapunov_multiplier(depth, exponent)
    perturbed = probability + (perturbation * multiplier)
    return max(0.01, min(0.99, perturbed))


def compute_divergence(prob_a: float, prob_b: float) -> float:
    """Kullback-Leibler inspired divergence measure between two probabilities.
    Returns a value between 0 (identical) and 1 (maximally divergent)."""
    p = max(0.001, min(0.999, prob_a))
    q = max(0.001, min(0.999, prob_b))

    # symmetric KL divergence, normalized to [0, 1]
    kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    symmetric_kl = (kl_pq + kl_qp) / 2.0

    # normalize using tanh to squash into [0, 1]
    return math.tanh(symmetric_kl)


def create_perturbed_graph(original: CausalGraph,
                           perturbation: float,
                           seed_mgr: SeedManager,
                           context: SituationContext | None = None) -> CausalGraph:
    import copy
    perturbed = copy.deepcopy(original)

    for node in perturbed.get_all_nodes():
        if node.node_id == perturbed.root_id:
            continue

        node.probability = apply_perturbation(
            node.probability, perturbation, node.depth, context
        )

    _renormalize_siblings(perturbed)

    return perturbed


def _renormalize_siblings(cg: CausalGraph) -> None:
    for node_id in cg.graph.nodes:
        children = cg.get_children(node_id)
        if not children:
            continue
        total = sum(c.probability for c in children)
        if total > 0:
            parent = cg.get_node(node_id)
            for child in children:
                child.probability = (child.probability / total) * parent.probability
                edge = cg.get_edge(node_id, child.node_id)
                edge.probability = child.probability / parent.probability if parent.probability > 0 else 0.0


def compute_graph_divergence(graph_a: CausalGraph,
                             graph_b: CausalGraph) -> float:
    """Compute the overall divergence between two causal graphs by
    comparing the probability distributions of their leaf nodes."""
    leaves_a = {n.node_id: n.probability for n in graph_a.get_leaves()}
    leaves_b = {n.node_id: n.probability for n in graph_b.get_leaves()}

    all_ids = set(leaves_a.keys()) | set(leaves_b.keys())
    if not all_ids:
        return 0.0

    divergences = []
    for node_id in all_ids:
        pa = leaves_a.get(node_id, 0.0)
        pb = leaves_b.get(node_id, 0.0)
        divergences.append(compute_divergence(max(pa, 0.001), max(pb, 0.001)))

    return sum(divergences) / len(divergences)


def fate_divergence_score(graph_a: CausalGraph,
                          graph_b: CausalGraph) -> float:
    """Human-readable divergence score as a percentage.
    0% = identical futures. 100% = completely different futures."""
    raw = compute_graph_divergence(graph_a, graph_b)
    return round(raw * 100, 1)
