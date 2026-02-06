from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

from simulus.core.parser import (
    SituationContext, ActorProfile, PowerDynamic, Precondition, ConflictVector,
)
from simulus.seed import SeedManager


class NodeType(Enum):
    ROOT = "root"
    DECISION = "decision"
    CONSEQUENCE = "consequence"
    OUTCOME = "outcome"
    ACTOR_REACTION = "actor_reaction"
    FEEDBACK = "feedback"


class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class CausalNode:
    node_id: str
    label: str
    node_type: NodeType
    depth: int
    probability: float = 1.0
    sentiment: Sentiment = Sentiment.NEUTRAL
    description: str = ""
    actor: str = "self"
    causal_mechanism: str = ""


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    probability: float
    label: str = ""
    edge_type: str = "causal"


@dataclass
class CausalGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    root_id: str = ""
    max_depth: int = 6

    def add_node(self, node: CausalNode) -> None:
        self.graph.add_node(node.node_id, data=node)

    def add_edge(self, edge: CausalEdge) -> None:
        self.graph.add_edge(edge.source_id, edge.target_id, data=edge)

    def get_node(self, node_id: str) -> CausalNode:
        return self.graph.nodes[node_id]["data"]

    def get_children(self, node_id: str) -> list[CausalNode]:
        children_ids = list(self.graph.successors(node_id))
        return [self.get_node(cid) for cid in children_ids]

    def get_edge(self, source_id: str, target_id: str) -> CausalEdge:
        return self.graph.edges[source_id, target_id]["data"]

    def get_leaves(self) -> list[CausalNode]:
        return [self.get_node(n) for n in self.graph.nodes
                if self.graph.out_degree(n) == 0]

    def get_all_nodes(self) -> list[CausalNode]:
        return [self.get_node(n) for n in self.graph.nodes]

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def get_feedback_edges(self) -> list[CausalEdge]:
        return [self.graph.edges[u, v]["data"]
                for u, v in self.graph.edges
                if self.graph.edges[u, v]["data"].edge_type == "feedback"]


# contextual consequence templates keyed by domain
# each template references the situation rather than being generic
CONTEXTUAL_CONSEQUENCES: dict[str, dict[str, list[dict]]] = {
    "career": {
        "positive": [
            {"label": "Reputation strengthens at work", "sentiment": Sentiment.POSITIVE,
             "mechanism": "social proof from bold action"},
            {"label": "New professional opportunities emerge", "sentiment": Sentiment.POSITIVE,
             "mechanism": "visibility leads to offers"},
            {"label": "Financial position improves", "sentiment": Sentiment.POSITIVE,
             "mechanism": "risk-reward payoff"},
            {"label": "Work-life balance shifts unexpectedly", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "career change has side effects"},
        ],
        "negative": [
            {"label": "Professional relationships strained", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "action creates workplace friction"},
            {"label": "Income stability threatened", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "career risk materializes"},
            {"label": "Forced to recalibrate career plan", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "setback requires adaptation"},
            {"label": "Learn from the setback", "sentiment": Sentiment.POSITIVE,
             "mechanism": "failure teaches resilience"},
        ],
        "neutral": [
            {"label": "Workplace dynamics subtly shift", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "action has indirect social effects"},
            {"label": "A colleague notices and responds", "sentiment": Sentiment.POSITIVE,
             "mechanism": "social observation triggers reaction"},
            {"label": "Quiet resentment builds among peers", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "envy or distrust accumulates"},
            {"label": "A new decision point at work approaches", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "consequences create new choices"},
        ],
    },
    "relationship": {
        "positive": [
            {"label": "Trust deepens between you", "sentiment": Sentiment.POSITIVE,
             "mechanism": "vulnerability builds connection"},
            {"label": "Emotional intimacy grows", "sentiment": Sentiment.POSITIVE,
             "mechanism": "honesty strengthens bond"},
            {"label": "Expectations change on both sides", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "new dynamic requires adjustment"},
            {"label": "Old wounds resurface unexpectedly", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "closeness exposes unresolved pain"},
        ],
        "negative": [
            {"label": "Communication breaks down further", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "hurt triggers withdrawal"},
            {"label": "One of you seeks outside support", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "stress drives need for external perspective"},
            {"label": "A fundamental incompatibility surfaces", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "conflict reveals deeper issues"},
            {"label": "Empathy emerges from shared struggle", "sentiment": Sentiment.POSITIVE,
             "mechanism": "adversity can create understanding"},
        ],
        "neutral": [
            {"label": "Relationship enters a holding pattern", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "neither party escalates or resolves"},
            {"label": "Small gestures shift the mood", "sentiment": Sentiment.POSITIVE,
             "mechanism": "micro-interactions accumulate"},
            {"label": "Unspoken tension quietly grows", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "avoidance compounds stress"},
            {"label": "A third party influences the dynamic", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "external actor enters the equation"},
        ],
    },
    "health": {
        "positive": [
            {"label": "Early intervention pays off", "sentiment": Sentiment.POSITIVE,
             "mechanism": "proactive health decision succeeds"},
            {"label": "Mental clarity and energy improve", "sentiment": Sentiment.POSITIVE,
             "mechanism": "health gains cascade to wellbeing"},
            {"label": "Treatment side effects appear", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "medical intervention has costs"},
            {"label": "Lifestyle adjustment becomes routine", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "new normal establishes itself"},
        ],
        "negative": [
            {"label": "Condition progresses without treatment", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "delay allows deterioration"},
            {"label": "Anxiety about health intensifies", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "uncertainty feeds worry"},
            {"label": "Body adapts and compensates", "sentiment": Sentiment.POSITIVE,
             "mechanism": "natural resilience activates"},
            {"label": "Seek a second opinion", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "uncertainty drives information seeking"},
        ],
        "neutral": [
            {"label": "Health remains stable but uncertain", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "no change is not resolution"},
            {"label": "A new symptom changes the picture", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "biological complexity introduces surprise"},
            {"label": "Support network strengthens around you", "sentiment": Sentiment.POSITIVE,
             "mechanism": "health concern activates care"},
            {"label": "Daily routine adapts to new reality", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "gradual normalization"},
        ],
    },
    "finance": {
        "positive": [
            {"label": "Investment yields early returns", "sentiment": Sentiment.POSITIVE,
             "mechanism": "market conditions favor position"},
            {"label": "Financial confidence grows", "sentiment": Sentiment.POSITIVE,
             "mechanism": "positive outcome reinforces strategy"},
            {"label": "Market volatility creates anxiety", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "gains are never secure"},
            {"label": "Lifestyle inflation tempts", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "wealth creates new spending pressure"},
        ],
        "negative": [
            {"label": "Losses compound under pressure", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "panic selling or doubling down"},
            {"label": "Debt restructuring becomes necessary", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "financial stress requires intervention"},
            {"label": "Find an unexpected income source", "sentiment": Sentiment.POSITIVE,
             "mechanism": "necessity drives creativity"},
            {"label": "Reassess financial priorities", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "loss forces honest evaluation"},
        ],
        "neutral": [
            {"label": "Markets move sideways", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "stagnation tests patience"},
            {"label": "A new opportunity appears in the noise", "sentiment": Sentiment.POSITIVE,
             "mechanism": "attention reveals hidden options"},
            {"label": "Inflation silently erodes position", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "inaction has hidden costs"},
            {"label": "Financial decision deadline approaches", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "time pressure forces action"},
        ],
    },
    "education": {
        "positive": [
            {"label": "Skill mastery accelerates", "sentiment": Sentiment.POSITIVE,
             "mechanism": "compounding knowledge effect"},
            {"label": "Mentor recognizes your potential", "sentiment": Sentiment.POSITIVE,
             "mechanism": "visibility attracts support"},
            {"label": "Burnout risk increases with intensity", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "effort has diminishing returns"},
            {"label": "New intellectual direction emerges", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "learning opens unexpected paths"},
        ],
        "negative": [
            {"label": "Confidence in abilities wavers", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "setback triggers self-doubt"},
            {"label": "Fall behind peers in progress", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "relative position declines"},
            {"label": "Discover a more suitable path", "sentiment": Sentiment.POSITIVE,
             "mechanism": "failure redirects toward fit"},
            {"label": "Take a strategic pause to regroup", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "rest enables future performance"},
        ],
        "neutral": [
            {"label": "Progress continues at steady pace", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "consistency yields gradual gains"},
            {"label": "An unexpected collaboration forms", "sentiment": Sentiment.POSITIVE,
             "mechanism": "proximity creates partnership"},
            {"label": "Institutional barriers slow progress", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "bureaucracy impedes goals"},
            {"label": "A crossroads in the academic path", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "completion creates new decisions"},
        ],
    },
    "travel": {
        "positive": [
            {"label": "New environment unlocks perspective", "sentiment": Sentiment.POSITIVE,
             "mechanism": "novelty stimulates growth"},
            {"label": "Meaningful connection in new place", "sentiment": Sentiment.POSITIVE,
             "mechanism": "openness attracts community"},
            {"label": "Homesickness weighs on the mind", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "distance has emotional cost"},
            {"label": "Identity shifts with the landscape", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "environment shapes self-concept"},
        ],
        "negative": [
            {"label": "Isolation deepens without support", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "disconnection from familiar network"},
            {"label": "Practical challenges accumulate", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "logistics of new life create friction"},
            {"label": "Resourcefulness grows from necessity", "sentiment": Sentiment.POSITIVE,
             "mechanism": "adversity builds capability"},
            {"label": "Consider returning or changing plans", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "reality tests commitment"},
        ],
        "neutral": [
            {"label": "Settling into an unfamiliar rhythm", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "adaptation is gradual"},
            {"label": "Serendipitous encounter changes plans", "sentiment": Sentiment.POSITIVE,
             "mechanism": "openness to chance"},
            {"label": "Cultural friction creates discomfort", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "values clash in new context"},
            {"label": "The next destination calls", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "restlessness or opportunity"},
        ],
    },
    "general": {
        "positive": [
            {"label": "Momentum builds from success", "sentiment": Sentiment.POSITIVE,
             "mechanism": "positive feedback loop"},
            {"label": "New possibilities become visible", "sentiment": Sentiment.POSITIVE,
             "mechanism": "action creates optionality"},
            {"label": "Unintended side effect emerges", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "complex systems produce surprises"},
            {"label": "Situation stabilizes at new level", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "system finds new equilibrium"},
        ],
        "negative": [
            {"label": "Pressure intensifies from setback", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "negative outcome compounds stress"},
            {"label": "Adapt strategy based on failure", "sentiment": Sentiment.POSITIVE,
             "mechanism": "learning from negative outcome"},
            {"label": "Situation continues to deteriorate", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "inertia of negative trajectory"},
            {"label": "Unexpected ally appears", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "adversity reveals hidden support"},
        ],
        "neutral": [
            {"label": "Life continues on its current path", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "no significant perturbation"},
            {"label": "A subtle shift in perspective", "sentiment": Sentiment.POSITIVE,
             "mechanism": "reflection creates insight"},
            {"label": "Quiet dissatisfaction accumulates", "sentiment": Sentiment.NEGATIVE,
             "mechanism": "unaddressed tension grows"},
            {"label": "A fork in the road approaches", "sentiment": Sentiment.NEUTRAL,
             "mechanism": "time creates new decision points"},
        ],
    },
}

# domain-specific decisions that reference the action context
CONTEXTUAL_DECISIONS: dict[str, list[dict]] = {
    "career": [
        {"label": "Take the bold professional leap",
         "description": "Commit fully to the career-changing action",
         "risk_level": 0.7, "reward_potential": 0.8},
        {"label": "Negotiate a safer middle path",
         "description": "Seek a compromise that limits downside",
         "risk_level": 0.3, "reward_potential": 0.5},
    ],
    "relationship": [
        {"label": "Address it honestly and directly",
         "description": "Confront the situation with transparency",
         "risk_level": 0.5, "reward_potential": 0.7},
        {"label": "Give it time and space",
         "description": "Allow the situation to evolve naturally",
         "risk_level": 0.3, "reward_potential": 0.4},
    ],
    "health": [
        {"label": "Act on it immediately",
         "description": "Take proactive health measures now",
         "risk_level": 0.3, "reward_potential": 0.7},
        {"label": "Monitor and wait for clarity",
         "description": "Gather more information before acting",
         "risk_level": 0.5, "reward_potential": 0.4},
    ],
    "finance": [
        {"label": "Commit capital to the opportunity",
         "description": "Take the financial risk for potential gain",
         "risk_level": 0.7, "reward_potential": 0.8},
        {"label": "Preserve capital and wait",
         "description": "Protect existing resources",
         "risk_level": 0.2, "reward_potential": 0.3},
    ],
    "education": [
        {"label": "Pursue the ambitious path",
         "description": "Push beyond comfort zone for growth",
         "risk_level": 0.5, "reward_potential": 0.7},
        {"label": "Consolidate current position",
         "description": "Strengthen foundations before advancing",
         "risk_level": 0.2, "reward_potential": 0.4},
    ],
    "travel": [
        {"label": "Go -- commit to the change",
         "description": "Embrace the uncertainty of relocation",
         "risk_level": 0.6, "reward_potential": 0.7},
        {"label": "Stay and build where you are",
         "description": "Invest in the known environment",
         "risk_level": 0.2, "reward_potential": 0.5},
    ],
    "general": [
        {"label": "Act decisively now",
         "description": "Commit to the action",
         "risk_level": 0.5, "reward_potential": 0.6},
        {"label": "Deliberate and prepare further",
         "description": "Invest more time before committing",
         "risk_level": 0.2, "reward_potential": 0.4},
    ],
}

# actor reaction templates: how other actors respond to the main actor's decisions
ACTOR_REACTION_TEMPLATES: dict[str, list[dict]] = {
    "superior": [
        {"label": "{actor} approves and supports", "sentiment": Sentiment.POSITIVE,
         "mechanism": "authority endorses action"},
        {"label": "{actor} resists and pushes back", "sentiment": Sentiment.NEGATIVE,
         "mechanism": "authority opposes action"},
        {"label": "{actor} watches cautiously", "sentiment": Sentiment.NEUTRAL,
         "mechanism": "authority waits to judge results"},
    ],
    "equal": [
        {"label": "{actor} joins and collaborates", "sentiment": Sentiment.POSITIVE,
         "mechanism": "peer aligns with action"},
        {"label": "{actor} distances and withdraws", "sentiment": Sentiment.NEGATIVE,
         "mechanism": "peer disagrees with direction"},
        {"label": "{actor} has a mixed reaction", "sentiment": Sentiment.NEUTRAL,
         "mechanism": "peer is ambivalent"},
    ],
    "subordinate": [
        {"label": "{actor} follows your lead", "sentiment": Sentiment.POSITIVE,
         "mechanism": "dependent trusts decision"},
        {"label": "{actor} is affected and anxious", "sentiment": Sentiment.NEGATIVE,
         "mechanism": "dependent absorbs the risk"},
        {"label": "{actor} adapts to the new situation", "sentiment": Sentiment.NEUTRAL,
         "mechanism": "dependent adjusts"},
    ],
}


def _generate_node_id(depth: int, branch: int, sub: int = 0) -> str:
    return f"d{depth}_b{branch}_s{sub}"


def _pick_contextual_consequences(domain: str, sentiment_key: str,
                                  seed_mgr: SeedManager,
                                  count: int = 2) -> list[dict]:
    domain_consequences = CONTEXTUAL_CONSEQUENCES.get(
        domain, CONTEXTUAL_CONSEQUENCES["general"])
    chain = domain_consequences.get(sentiment_key, domain_consequences["neutral"])
    result = []
    for _ in range(count):
        idx = seed_mgr.integers(0, len(chain))
        result.append(chain[idx])
    return result


def _generate_actor_reaction(actor: ActorProfile,
                             parent_sentiment: Sentiment,
                             seed_mgr: SeedManager) -> dict | None:
    power_key = {
        PowerDynamic.SUPERIOR: "superior",
        PowerDynamic.EQUAL: "equal",
        PowerDynamic.SUBORDINATE: "subordinate",
        PowerDynamic.UNKNOWN: "equal",
    }[actor.power_dynamic]

    templates = ACTOR_REACTION_TEMPLATES.get(power_key, ACTOR_REACTION_TEMPLATES["equal"])

    # bias reaction based on parent sentiment
    if parent_sentiment == Sentiment.POSITIVE:
        weights = [0.5, 0.2, 0.3]
    elif parent_sentiment == Sentiment.NEGATIVE:
        weights = [0.2, 0.5, 0.3]
    else:
        weights = [0.33, 0.33, 0.34]

    idx = seed_mgr.choice(list(range(len(templates))), p=weights)
    template = templates[idx]
    return {
        "label": template["label"].format(actor=actor.name.title()),
        "sentiment": template["sentiment"],
        "mechanism": template["mechanism"],
        "actor": actor.name,
    }


def _apply_precondition_modifiers(base_prob: float,
                                  preconditions: list[Precondition],
                                  seed_mgr: SeedManager) -> float:
    modifier = 0.0
    for pc in preconditions:
        modifier += pc.sentiment_impact * 0.1
        modifier += pc.trust_modifier * 0.05
    noise = seed_mgr.normal(0.0, 0.01)
    return max(0.05, min(0.95, base_prob + modifier + noise))


def build_causal_graph(context: SituationContext, seed_mgr: SeedManager,
                       max_depth: int = 6) -> CausalGraph:
    cg = CausalGraph(max_depth=max_depth)

    root = CausalNode(
        node_id="root",
        label=context.raw_input,
        node_type=NodeType.ROOT,
        depth=0,
        description=context.summary,
    )
    cg.add_node(root)
    cg.root_id = root.node_id

    decisions = CONTEXTUAL_DECISIONS.get(context.domain, CONTEXTUAL_DECISIONS["general"])
    total_decisions = len(decisions)
    base_probs = []
    remaining = 1.0
    for i in range(total_decisions):
        if i == total_decisions - 1:
            base_probs.append(remaining)
        else:
            p = seed_mgr.uniform(0.25, 0.75) * remaining
            base_probs.append(p)
            remaining -= p

    # apply precondition modifiers to decision probabilities
    if context.preconditions:
        for i in range(len(base_probs)):
            base_probs[i] = _apply_precondition_modifiers(
                base_probs[i], context.preconditions, seed_mgr)
        total = sum(base_probs)
        base_probs = [p / total for p in base_probs]

    for branch_idx, decision in enumerate(decisions):
        branch_seed = seed_mgr.fork(f"branch_{branch_idx}")
        d_node = CausalNode(
            node_id=_generate_node_id(1, branch_idx),
            label=decision["label"],
            node_type=NodeType.DECISION,
            depth=1,
            probability=base_probs[branch_idx],
            description=decision.get("description", ""),
            causal_mechanism=f"risk={decision.get('risk_level', 0.5):.1f}, reward={decision.get('reward_potential', 0.5):.1f}",
        )
        cg.add_node(d_node)
        cg.add_edge(CausalEdge(
            source_id=root.node_id,
            target_id=d_node.node_id,
            probability=base_probs[branch_idx],
            label=decision["label"],
        ))

        _expand_branch(cg, d_node, decision, branch_seed, branch_idx,
                       max_depth, context)

    # add feedback edges for structural causal model
    _add_feedback_edges(cg, context)

    return cg


def _expand_branch(cg: CausalGraph, parent: CausalNode, decision: dict,
                   seed_mgr: SeedManager, branch_idx: int,
                   max_depth: int, context: SituationContext) -> None:
    current_nodes = [parent]

    for depth in range(2, max_depth + 1):
        next_nodes = []
        for sub_idx, current in enumerate(current_nodes):
            current_sentiment = current.sentiment.value

            # pick contextual consequences from the domain-specific pool
            consequences = _pick_contextual_consequences(
                context.domain, current_sentiment, seed_mgr, count=2)

            p_split = seed_mgr.uniform(0.3, 0.7)
            probs = [p_split, 1.0 - p_split]

            for cons_idx, cons in enumerate(consequences):
                node_id = _generate_node_id(depth, branch_idx * 100 + sub_idx, cons_idx)

                if depth == max_depth:
                    node_type = NodeType.OUTCOME
                else:
                    node_type = NodeType.CONSEQUENCE

                node = CausalNode(
                    node_id=node_id,
                    label=cons["label"],
                    node_type=node_type,
                    depth=depth,
                    probability=current.probability * probs[cons_idx],
                    sentiment=cons["sentiment"],
                    causal_mechanism=cons.get("mechanism", ""),
                )
                cg.add_node(node)
                cg.add_edge(CausalEdge(
                    source_id=current.node_id,
                    target_id=node.node_id,
                    probability=probs[cons_idx],
                    label=cons["label"],
                ))
                next_nodes.append(node)

            # inject actor reactions at depth 3 for each known actor
            if depth == 3 and context.actor_profiles:
                for actor_idx, actor in enumerate(context.actor_profiles[:2]):
                    reaction = _generate_actor_reaction(
                        actor, current.sentiment, seed_mgr)
                    if reaction is None:
                        continue

                    reaction_id = f"d{depth}_b{branch_idx}_actor_{actor_idx}_s{sub_idx}"
                    reaction_prob = seed_mgr.uniform(0.05, 0.15)

                    # steal probability from existing children proportionally
                    for existing in next_nodes[-len(consequences):]:
                        existing.probability *= (1.0 - reaction_prob / 2)

                    reaction_node = CausalNode(
                        node_id=reaction_id,
                        label=reaction["label"],
                        node_type=NodeType.ACTOR_REACTION,
                        depth=depth,
                        probability=current.probability * reaction_prob,
                        sentiment=reaction["sentiment"],
                        actor=reaction.get("actor", "other"),
                        causal_mechanism=reaction.get("mechanism", ""),
                    )
                    cg.add_node(reaction_node)
                    cg.add_edge(CausalEdge(
                        source_id=current.node_id,
                        target_id=reaction_node.node_id,
                        probability=reaction_prob,
                        label=reaction["label"],
                        edge_type="actor_reaction",
                    ))
                    next_nodes.append(reaction_node)

        current_nodes = next_nodes


def _add_feedback_edges(cg: CausalGraph, context: SituationContext) -> None:
    """Add feedback edges that model second-order effects.
    For example, financial stress feeds back into relationship quality,
    health affects career performance, etc.

    These edges don't change the DAG structure (we don't create cycles)
    but they are stored as metadata for the Bayesian engine to use
    when computing conditional probabilities."""

    feedback_rules = {
        "career": [
            ("financial stress", "relationship strain", -0.15),
            ("professional success", "self-confidence", 0.12),
            ("workplace conflict", "mental health", -0.10),
        ],
        "relationship": [
            ("emotional distress", "work performance", -0.12),
            ("relationship stability", "mental health", 0.15),
            ("trust breakdown", "self-worth", -0.18),
        ],
        "health": [
            ("health anxiety", "relationship strain", -0.10),
            ("physical recovery", "career capacity", 0.12),
            ("chronic condition", "financial pressure", -0.15),
        ],
        "finance": [
            ("financial loss", "relationship tension", -0.18),
            ("financial security", "mental health", 0.12),
            ("debt pressure", "career desperation", -0.10),
        ],
        "education": [
            ("academic stress", "social life", -0.10),
            ("academic success", "career prospects", 0.15),
            ("burnout", "mental health", -0.15),
        ],
        "travel": [
            ("isolation", "mental health", -0.12),
            ("new community", "self-growth", 0.15),
            ("cultural shock", "relationship with home", -0.10),
        ],
    }

    rules = feedback_rules.get(context.domain, [])
    all_nodes = cg.get_all_nodes()

    for source_pattern, target_pattern, strength in rules:
        source_nodes = [n for n in all_nodes
                        if source_pattern.lower() in n.label.lower()
                        or source_pattern.lower() in n.causal_mechanism.lower()]
        target_nodes = [n for n in all_nodes
                        if target_pattern.lower() in n.label.lower()
                        or target_pattern.lower() in n.causal_mechanism.lower()]

        # connect the first matching pair if they exist at different depths
        for sn in source_nodes[:1]:
            for tn in target_nodes[:1]:
                if sn.depth < tn.depth and sn.node_id != tn.node_id:
                    cg.add_edge(CausalEdge(
                        source_id=sn.node_id,
                        target_id=tn.node_id,
                        probability=abs(strength),
                        label=f"feedback: {source_pattern} -> {target_pattern}",
                        edge_type="feedback",
                    ))
