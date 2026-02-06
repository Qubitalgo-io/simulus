from __future__ import annotations

from simulus.core.causal_graph import CausalGraph, CausalNode, Sentiment, NodeType
from simulus.core.montecarlo import SimulationResult
from simulus.core.parser import SituationContext


def _most_probable_path(cg: CausalGraph) -> list[CausalNode]:
    path = []
    current_id = cg.root_id
    while True:
        node = cg.get_node(current_id)
        path.append(node)
        children = cg.get_children(current_id)
        if not children:
            break
        best = max(children, key=lambda c: c.probability)
        current_id = best.node_id
    return path


def _sentiment_word(score: float) -> str:
    if score > 0.3:
        return "strongly favorable"
    if score > 0.1:
        return "cautiously optimistic"
    if score > -0.1:
        return "uncertain, with nearly equal chances of good and bad outcomes"
    if score > -0.3:
        return "tilted toward difficulty"
    return "weighted toward hardship"


def _outcome_summary(mc_result: SimulationResult, top_n: int = 3) -> str:
    top = mc_result.top_outcomes(top_n)
    if not top:
        return "No clear outcome emerged."

    parts = []
    for label, prob in top:
        pct = prob * 100
        parts.append(f"{label} ({pct:.0f}%)")

    if len(parts) == 1:
        return f"The most likely outcome is: {parts[0]}."
    return ("The most likely outcomes are: " +
            ", ".join(parts[:-1]) + f", and {parts[-1]}.")


def _path_narrative(path: list[CausalNode]) -> str:
    if len(path) < 3:
        return ""

    decision = path[1] if len(path) > 1 else None
    midpoints = path[2:-1]
    endpoint = path[-1] if len(path) > 1 else None

    parts = []
    if decision:
        parts.append(f'The most probable path begins with "{decision.label.lower()}"')

    if midpoints:
        mid_labels = [f'"{n.label.lower()}"' for n in midpoints[:2]]
        parts.append(", leading through " + " and then ".join(mid_labels))

    if endpoint:
        parts.append(f', ultimately arriving at "{endpoint.label.lower()}"')

    return "".join(parts) + "."


def _volatility_note(context: SituationContext) -> str:
    v = context.compound_volatility
    if v > 0.7:
        return ("This situation is highly volatile. Small changes in timing or "
                "attitude could radically alter the outcome.")
    if v > 0.4:
        return ("This situation carries moderate volatility. The outcome is "
                "sensitive to how events unfold in the near term.")
    return ("This situation is relatively stable. The outcome is less "
            "sensitive to small perturbations.")


def _actor_note(context: SituationContext) -> str:
    if not context.actor_profiles:
        return ""

    actors = context.actor_profiles[:3]
    if len(actors) == 1:
        a = actors[0]
        dynamic = a.power_dynamic.value
        if dynamic == "superior":
            return (f"{a.name.title()}, who holds authority in this situation, "
                    "has significant influence over how this unfolds. "
                    "Their reaction is a key variable.")
        elif dynamic == "subordinate":
            return (f"{a.name.title()} is affected by your decision but has "
                    "limited power to change the trajectory.")
        else:
            return (f"{a.name.title()} is an equal participant in this situation. "
                    "The outcome depends heavily on both of your choices.")

    names = [a.name.title() for a in actors]
    return ("Multiple actors shape this scenario: " +
            ", ".join(names[:-1]) + f" and {names[-1]}. " +
            "Their individual reactions introduce additional uncertainty.")


def _divergence_note(butterfly_divergence: float | None) -> str:
    if butterfly_divergence is None:
        return ""
    if butterfly_divergence < 1.0:
        return ("A small perturbation barely changes the outcome -- this "
                "trajectory is relatively stable against minor variations.")
    if butterfly_divergence < 10.0:
        return ("A small perturbation produces noticeable divergence. "
                "This scenario exhibits sensitive dependence on initial conditions.")
    return ("Even a tiny change produces wildly different outcomes. "
            "This is a textbook case of sensitive dependence -- the "
            "trajectories diverge exponentially with depth.")


def generate_explanation(context: SituationContext,
                         cg: CausalGraph,
                         mc_result: SimulationResult,
                         sentiment_score: float,
                         butterfly_divergence: float | None = None) -> str:
    sections = []

    # opening
    domain_labels = {
        "career": "a career decision",
        "relationship": "a relationship crossroads",
        "health": "a health-related choice",
        "finance": "a financial decision",
        "education": "an educational turning point",
        "travel": "a life relocation decision",
        "general": "a significant life decision",
    }
    domain_label = domain_labels.get(context.domain, "a significant decision")
    sections.append(f"This is {domain_label}.")

    # sentiment summary
    sentiment_desc = _sentiment_word(sentiment_score)
    pos_pct = mc_result.sentiment_distribution.get("positive", 0) * 100
    neg_pct = mc_result.sentiment_distribution.get("negative", 0) * 100
    sections.append(
        f"Across {mc_result.n_simulations:,} simulated scenario branches, the overall "
        f"outlook is {sentiment_desc} -- {pos_pct:.0f}% of outcomes are "
        f"positive and {neg_pct:.0f}% are negative."
    )

    # most probable path
    path = _most_probable_path(cg)
    path_text = _path_narrative(path)
    if path_text:
        sections.append(path_text)

    # top outcomes
    outcome_text = _outcome_summary(mc_result)
    sections.append(outcome_text)

    # actor dynamics
    actor_text = _actor_note(context)
    if actor_text:
        sections.append(actor_text)

    # volatility
    sections.append(_volatility_note(context))

    # butterfly effect
    if butterfly_divergence is not None:
        div_text = _divergence_note(butterfly_divergence)
        if div_text:
            sections.append(div_text)

    # closing
    if sentiment_score > 0.1:
        closing = ("The probabilities lean in your favor, but sensitive "
                   "dependence on initial conditions means a small change "
                   "could reshape everything. The seed is set -- but you "
                   "cannot see all the variables.")
    elif sentiment_score < -0.1:
        closing = ("The probabilities suggest a difficult road ahead. "
                   "That does not make it the wrong road -- only the harder one. "
                   "The seed is set -- but you cannot see all the variables.")
    else:
        closing = ("The trajectories balance on a knife edge. Neither path clearly "
                   "dominates. The seed is set -- but you cannot see all the variables.")

    sections.append(closing)

    return " ".join(sections)
