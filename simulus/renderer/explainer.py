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


def _conflict_note(context: SituationContext) -> str:
    if not context.conflict_vectors:
        return ""

    cv = context.conflict_vectors[0]
    a = cv.actor_a.title() if cv.actor_a != "self" else "you"
    b = cv.actor_b.title() if cv.actor_b != "self" else "you"

    if cv.actor_a == "self" or cv.actor_b == "self":
        other = b if cv.actor_a == "self" else a
        nature = cv.nature.lower()
        if cv.intensity > 0.6:
            return (f"There is a pronounced tension between {other} and "
                    f"your own position: {nature}. This internal friction "
                    "amplifies the uncertainty at every branch point.")
        return (f"A conflict exists between {other} and your stance: "
                f"{nature}. This shapes how the branches diverge.")

    if cv.intensity > 0.6:
        return (f"A high-intensity conflict between {a} and {b} -- "
                f"{cv.nature.lower()} -- adds significant unpredictability.")
    return (f"There is a conflict between {a} and {b}: "
            f"{cv.nature.lower()}.")


def _stakes_note(context: SituationContext) -> str:
    if not context.stakes:
        return ""

    named = [s for s in context.stakes[:3]]
    severity = context.stake_severity

    if severity > 0.7:
        qualifier = "The stakes are high"
    elif severity > 0.4:
        qualifier = "The stakes are moderate"
    else:
        qualifier = "The stakes are relatively low"

    if len(named) == 1:
        return f"{qualifier}: {named[0].lower()} is on the line."
    return (f"{qualifier}: {', '.join(s.lower() for s in named[:-1])} "
            f"and {named[-1].lower()} are all on the line.")


def _emotional_context_note(context: SituationContext) -> str:
    ml = context.ml_signals
    emotion = context.emotional_state

    if ml and "emotion_distribution" in ml:
        dist = ml["emotion_distribution"]
        confidence = ml.get("emotion_confidence", 0.0)

        if not dist:
            return _simple_emotion_note(emotion)

        sorted_emotions = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_prob = sorted_emotions[0]

        if len(sorted_emotions) > 1:
            second_emotion, second_prob = sorted_emotions[1]
        else:
            return _simple_emotion_note(emotion)

        if confidence < 0.5:
            return ("The emotional signal is ambiguous -- the model cannot "
                    "confidently classify the affect, which itself suggests "
                    "unresolved internal tension.")

        if second_prob > 0.2 and top_prob - second_prob < 0.3:
            return (f"The model reads the dominant affect as {top_emotion}, "
                    f"but there is a secondary signal of {second_emotion} "
                    f"({second_prob:.0%}). This mixed emotional state "
                    "introduces additional branching volatility.")

        return (f"The model reads the primary affect as {top_emotion} "
                f"(confidence: {top_prob:.0%}). This emotional baseline "
                "colors the probability weighting at each decision point.")

    return _simple_emotion_note(emotion)


def _simple_emotion_note(emotion: str) -> str:
    if emotion == "neutral":
        return ""

    descriptors = {
        "anxiety": "an undercurrent of anxiety, which tends to amplify worst-case branches",
        "hopeful": "a hopeful orientation, which weights toward opportunity-seeking paths",
        "fear": "a fearful baseline, which biases the model toward avoidance trajectories",
        "anger": "an angry affect, which increases the probability of confrontational branches",
        "sadness": "a melancholic tone, which shifts weight toward loss-related outcomes",
        "excited": "an excited energy, which amplifies both upside and downside extremes",
    }
    desc = descriptors.get(emotion)
    if desc:
        return f"The emotional context carries {desc}."
    return f"The detected emotional state is {emotion}."


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

    # situation-specific texture
    stakes_text = _stakes_note(context)
    if stakes_text:
        sections.append(stakes_text)

    conflict_text = _conflict_note(context)
    if conflict_text:
        sections.append(conflict_text)

    emotion_text = _emotional_context_note(context)
    if emotion_text:
        sections.append(emotion_text)

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
