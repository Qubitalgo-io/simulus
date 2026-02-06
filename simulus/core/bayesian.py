from __future__ import annotations

# NOTE: Despite the filename, this module does NOT implement Bayesian
# inference in the textbook sense.  There is no likelihood function and
# no evidence-conditioned posterior update.  Instead, it uses domain-
# specific sentiment priors, first-order Markov transition matrices, and
# a stack of additive heuristic modifiers (emotion, power, severity,
# feedback) to score each node's conditional sentiment probability.
# This is closer to a parameterized scoring model than to Bayes' theorem.
# The name is kept for backward compatibility with the rest of the codebase.

import numpy as np

from simulus.core.causal_graph import CausalGraph, CausalNode, Sentiment, NodeType
from simulus.core.parser import SituationContext, ActorProfile, PowerDynamic
from simulus.seed import SeedManager


SENTIMENT_PRIORS: dict[str, dict[str, float]] = {
    "career": {"positive": 0.45, "negative": 0.35, "neutral": 0.20},
    "relationship": {"positive": 0.40, "negative": 0.40, "neutral": 0.20},
    "health": {"positive": 0.50, "negative": 0.30, "neutral": 0.20},
    "finance": {"positive": 0.35, "negative": 0.45, "neutral": 0.20},
    "education": {"positive": 0.55, "negative": 0.25, "neutral": 0.20},
    "travel": {"positive": 0.50, "negative": 0.30, "neutral": 0.20},
    "general": {"positive": 0.40, "negative": 0.35, "neutral": 0.25},
}

TRANSITION_MATRIX: dict[str, dict[str, float]] = {
    "positive": {"positive": 0.55, "negative": 0.20, "neutral": 0.25},
    "negative": {"positive": 0.25, "negative": 0.50, "neutral": 0.25},
    "neutral":  {"positive": 0.35, "negative": 0.30, "neutral": 0.35},
}

# 2nd-order transitions: when grandparent and parent share a sentiment,
# a streak bonus amplifies continuation.  This models real-world momentum
# where sustained good/bad runs are self-reinforcing.
STREAK_BONUS: dict[str, dict[str, float]] = {
    "positive": {"positive": 0.10, "negative": -0.05, "neutral": -0.05},
    "negative": {"positive": -0.05, "negative": 0.10, "neutral": -0.05},
    "neutral":  {"positive": 0.00, "negative": 0.00, "neutral": 0.00},
}

# mean-reversion coefficients: extreme outcomes at deep levels tend
# toward the center.  This prevents runaway positive/negative spirals
# that are unrealistic in most real-life scenarios.
MEAN_REVERSION_STRENGTH = 0.08
MEAN_REVERSION_DEPTH_THRESHOLD = 3

EMOTION_MODIFIERS: dict[str, dict[str, float]] = {
    "anxious":    {"positive": -0.10, "negative": 0.10, "neutral": 0.00},
    "confident":  {"positive": 0.15, "negative": -0.10, "neutral": -0.05},
    "angry":      {"positive": -0.15, "negative": 0.15, "neutral": 0.00},
    "hopeful":    {"positive": 0.10, "negative": -0.05, "neutral": -0.05},
    "desperate":  {"positive": -0.05, "negative": 0.15, "neutral": -0.10},
    "neutral":    {"positive": 0.00, "negative": 0.00, "neutral": 0.00},
}

# power dynamics shift probabilities: superiors have more influence on outcomes
POWER_MODIFIERS: dict[str, dict[str, float]] = {
    "superior":    {"positive": -0.08, "negative": 0.08, "neutral": 0.00},
    "equal":       {"positive": 0.00, "negative": 0.00, "neutral": 0.00},
    "subordinate": {"positive": 0.05, "negative": -0.03, "neutral": -0.02},
    "unknown":     {"positive": 0.00, "negative": 0.00, "neutral": 0.00},
}

# severity multiplier: higher severity amplifies negative outcomes
SEVERITY_TRANSITION_SHIFT: dict[str, float] = {
    "positive": -0.05,
    "negative": 0.08,
    "neutral": -0.03,
}


def _normalize(probs: dict[str, float]) -> dict[str, float]:
    floored = {k: max(0.001, v) for k, v in probs.items()}
    total = sum(floored.values())
    return {k: v / total for k, v in floored.items()}


def _apply_emotion_modifier(priors: dict[str, float],
                            emotional_state: str) -> dict[str, float]:
    modifiers = EMOTION_MODIFIERS.get(emotional_state, EMOTION_MODIFIERS["neutral"])
    adjusted = {k: priors[k] + modifiers.get(k, 0.0) for k in priors}
    return _normalize(adjusted)


def _apply_power_modifier(priors: dict[str, float],
                          context: SituationContext | None) -> dict[str, float]:
    if context is None or not context.actor_profiles:
        return priors

    # aggregate influence of all actors weighted by power
    combined_mod = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for actor in context.actor_profiles[:3]:
        key = actor.power_dynamic.value
        mods = POWER_MODIFIERS.get(key, POWER_MODIFIERS["unknown"])
        for s in combined_mod:
            combined_mod[s] += mods[s]

    n_actors = min(len(context.actor_profiles), 3)
    adjusted = {k: priors[k] + combined_mod[k] / n_actors for k in priors}
    return _normalize(adjusted)


def _apply_severity_shift(priors: dict[str, float],
                          severity: float) -> dict[str, float]:
    if severity < 0.3:
        return priors
    scale = severity - 0.3
    adjusted = {k: priors[k] + SEVERITY_TRANSITION_SHIFT[k] * scale for k in priors}
    return _normalize(adjusted)


def _apply_feedback_modifier(priors: dict[str, float],
                             node: CausalNode,
                             cg: CausalGraph) -> dict[str, float]:
    feedback_edges = cg.get_feedback_edges()
    modifier = 0.0
    for edge in feedback_edges:
        if edge.target_id == node.node_id:
            source = cg.get_node(edge.source_id)
            if source.sentiment == Sentiment.NEGATIVE:
                modifier -= edge.probability * 0.5
            elif source.sentiment == Sentiment.POSITIVE:
                modifier += edge.probability * 0.5

    if abs(modifier) < 0.001:
        return priors

    adjusted = {
        "positive": priors["positive"] + modifier * 0.3,
        "negative": priors["negative"] - modifier * 0.3,
        "neutral": priors["neutral"],
    }
    return _normalize(adjusted)


def _apply_streak_bonus(priors: dict[str, float],
                        parent: CausalNode,
                        cg: CausalGraph) -> dict[str, float]:
    predecessors = list(cg.graph.predecessors(parent.node_id))
    if not predecessors:
        return priors

    grandparent = cg.get_node(predecessors[0])
    if grandparent.sentiment != parent.sentiment:
        return priors

    streak_key = parent.sentiment.value
    bonus = STREAK_BONUS.get(streak_key, STREAK_BONUS["neutral"])
    adjusted = {k: priors[k] + bonus[k] for k in priors}
    return _normalize(adjusted)


def _apply_mean_reversion(priors: dict[str, float],
                          depth: int) -> dict[str, float]:
    if depth < MEAN_REVERSION_DEPTH_THRESHOLD:
        return priors

    depth_factor = (depth - MEAN_REVERSION_DEPTH_THRESHOLD + 1) * MEAN_REVERSION_STRENGTH
    neutral_center = 1.0 / len(priors)
    adjusted = {}
    for k, v in priors.items():
        adjusted[k] = v + depth_factor * (neutral_center - v)
    return _normalize(adjusted)


def _apply_depth_uncertainty(priors: dict[str, float],
                             depth: int,
                             seed_mgr: SeedManager) -> dict[str, float]:
    if depth <= 2:
        return priors

    spread = 0.015 * (depth - 2)
    adjusted = {}
    for k, v in priors.items():
        noise = seed_mgr.normal(0.0, spread)
        adjusted[k] = v + noise
    return _normalize(adjusted)


def compute_node_probability(node: CausalNode, parent: CausalNode | None,
                             domain: str, emotional_state: str,
                             seed_mgr: SeedManager,
                             context: SituationContext | None = None,
                             cg: CausalGraph | None = None) -> float:
    if parent is None:
        priors = SENTIMENT_PRIORS.get(domain, SENTIMENT_PRIORS["general"])
        adjusted = _apply_emotion_modifier(priors, emotional_state)
        if context:
            adjusted = _apply_power_modifier(adjusted, context)
            adjusted = _apply_severity_shift(adjusted, context.stake_severity)
        return adjusted.get(node.sentiment.value, 0.33)

    parent_sentiment = parent.sentiment.value
    transition = TRANSITION_MATRIX.get(parent_sentiment, TRANSITION_MATRIX["neutral"])
    adjusted = _apply_emotion_modifier(transition, emotional_state)

    if context:
        adjusted = _apply_power_modifier(adjusted, context)
        adjusted = _apply_severity_shift(adjusted, context.stake_severity)

    if cg:
        adjusted = _apply_feedback_modifier(adjusted, node, cg)
        adjusted = _apply_streak_bonus(adjusted, parent, cg)

    adjusted = _apply_mean_reversion(adjusted, node.depth)
    adjusted = _apply_depth_uncertainty(adjusted, node.depth, seed_mgr)

    # actor reaction nodes get a modifier based on the actor's power dynamic
    if node.node_type == NodeType.ACTOR_REACTION and context:
        actor_match = [a for a in context.actor_profiles
                       if a.name.lower() in node.actor.lower()]
        if actor_match:
            actor = actor_match[0]
            power_key = actor.power_dynamic.value
            power_mod = POWER_MODIFIERS.get(power_key, POWER_MODIFIERS["unknown"])
            adjusted = {k: adjusted[k] + power_mod[k] * 0.5 for k in adjusted}
            adjusted = _normalize(adjusted)

    noise = seed_mgr.normal(0.0, 0.02)
    base_prob = adjusted.get(node.sentiment.value, 0.33) + noise
    return max(0.01, min(0.99, base_prob))


def update_graph_probabilities(cg: CausalGraph, domain: str,
                               emotional_state: str,
                               seed_mgr: SeedManager,
                               context: SituationContext | None = None) -> None:
    for node in cg.get_all_nodes():
        if node.node_id == cg.root_id:
            continue

        predecessors = list(cg.graph.predecessors(node.node_id))
        if not predecessors:
            continue
        parent = cg.get_node(predecessors[0])

        siblings = cg.get_children(parent.node_id)
        raw_probs = {}
        for sibling in siblings:
            p = compute_node_probability(sibling, parent, domain,
                                         emotional_state, seed_mgr,
                                         context=context, cg=cg)
            raw_probs[sibling.node_id] = p

        total = sum(raw_probs.values())
        if total > 0:
            for sibling in siblings:
                normalized_p = raw_probs[sibling.node_id] / total
                sibling.probability = (parent.probability if parent else 1.0) * normalized_p
                edge = cg.get_edge(parent.node_id, sibling.node_id)
                edge.probability = normalized_p

    _normalize_leaf_probabilities(cg)


def _normalize_leaf_probabilities(cg: CausalGraph) -> None:
    leaves = cg.get_leaves()
    if not leaves:
        return
    total = sum(leaf.probability for leaf in leaves)
    if total <= 0 or abs(total - 1.0) < 1e-9:
        return
    scale = 1.0 / total
    for leaf in leaves:
        leaf.probability *= scale


def compute_outcome_distribution(cg: CausalGraph) -> dict[str, float]:
    leaves = cg.get_leaves()
    distribution = {}
    total = sum(leaf.probability for leaf in leaves)
    for leaf in leaves:
        distribution[leaf.node_id] = leaf.probability / total if total > 0 else 0.0
    return distribution


def expected_sentiment_score(cg: CausalGraph) -> float:
    """Compute a weighted sentiment score across all outcomes.
    Positive = +1, neutral = 0, negative = -1."""
    sentiment_values = {
        Sentiment.POSITIVE: 1.0,
        Sentiment.NEGATIVE: -1.0,
        Sentiment.NEUTRAL: 0.0,
    }
    leaves = cg.get_leaves()
    total_prob = sum(leaf.probability for leaf in leaves)
    if total_prob == 0:
        return 0.0
    score = sum(leaf.probability * sentiment_values[leaf.sentiment] for leaf in leaves)
    return score / total_prob
