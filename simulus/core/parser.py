from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class PowerDynamic(Enum):
    SUPERIOR = "superior"
    EQUAL = "equal"
    SUBORDINATE = "subordinate"
    UNKNOWN = "unknown"


class Reversibility(Enum):
    IRREVERSIBLE = "irreversible"
    DIFFICULT = "difficult"
    MODERATE = "moderate"
    EASY = "easy"


@dataclass
class ActorProfile:
    name: str
    role: str = ""
    relationship_to_main: str = ""
    power_dynamic: PowerDynamic = PowerDynamic.UNKNOWN
    emotional_state: str = "neutral"
    stake_level: float = 0.5

    @property
    def is_authority(self) -> bool:
        return self.power_dynamic == PowerDynamic.SUPERIOR


@dataclass
class Precondition:
    description: str
    sentiment_impact: float = 0.0
    trust_modifier: float = 0.0
    urgency_modifier: float = 0.0


@dataclass
class ConflictVector:
    actor_a: str
    actor_b: str
    nature: str
    intensity: float = 0.5


@dataclass
class SituationContext:
    raw_input: str
    actors: list[str] = field(default_factory=list)
    actor_profiles: list[ActorProfile] = field(default_factory=list)
    main_actor: str = ""
    stakes: list[str] = field(default_factory=list)
    stake_severity: float = 0.5
    environment: str = ""
    emotional_state: str = "neutral"
    domain: str = "general"
    action_verb: str = ""
    object_of_action: str = ""
    preconditions: list[Precondition] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    time_pressure: float = 0.0
    reversibility: Reversibility = Reversibility.MODERATE
    conflict_vectors: list[ConflictVector] = field(default_factory=list)
    volatility: float = 0.5
    ml_signals: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        parts = [f"Actor: {self.main_actor}"]
        if self.action_verb:
            parts.append(f"Action: {self.action_verb}")
        if self.object_of_action:
            parts.append(f"Object: {self.object_of_action}")
        if self.environment:
            parts.append(f"Setting: {self.environment}")
        if self.emotional_state != "neutral":
            parts.append(f"Mood: {self.emotional_state}")
        if self.preconditions:
            parts.append(f"Preconditions: {len(self.preconditions)}")
        if self.conflict_vectors:
            parts.append(f"Conflicts: {len(self.conflict_vectors)}")
        if self.time_pressure > 0.5:
            parts.append("Time-pressured")
        return " | ".join(parts)

    @property
    def compound_volatility(self) -> float:
        """Volatility increases with more actors, higher stakes, more conflicts,
        and tighter time pressure."""
        base = self.volatility
        actor_factor = min(len(self.actor_profiles) * 0.1, 0.3)
        conflict_factor = min(len(self.conflict_vectors) * 0.15, 0.3)
        pressure_factor = self.time_pressure * 0.2
        return min(1.0, base + actor_factor + conflict_factor + pressure_factor)


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "career": ["job", "boss", "work", "office", "salary", "raise", "promotion",
               "quit", "hire", "fire", "resign", "interview", "meeting", "colleague",
               "manager", "career", "company", "startup", "business"],
    "relationship": ["love", "partner", "date", "marriage", "breakup", "divorce",
                     "friend", "family", "parent", "child", "argument", "trust",
                     "relationship", "girlfriend", "boyfriend", "spouse", "wedding"],
    "health": ["doctor", "hospital", "sick", "exercise", "diet", "surgery",
               "diagnosis", "medication", "therapy", "mental", "anxiety",
               "depression", "injury", "pain", "health"],
    "finance": ["money", "invest", "debt", "loan", "mortgage", "stock", "save",
                "spend", "budget", "bank", "crypto", "retire", "wealth", "broke"],
    "education": ["school", "university", "study", "exam", "degree", "learn",
                  "teacher", "student", "course", "graduate", "research", "thesis"],
    "travel": ["move", "travel", "city", "country", "abroad", "relocate",
               "flight", "destination", "migrate", "visa", "home"],
}

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anxious": ["worried", "nervous", "anxious", "scared", "afraid", "unsure",
                "hesitant", "terrified", "stressed", "panic"],
    "confident": ["confident", "sure", "certain", "ready", "determined",
                  "bold", "decisive", "strong"],
    "angry": ["angry", "furious", "upset", "frustrated", "annoyed", "mad",
              "irritated", "resentful"],
    "hopeful": ["hopeful", "optimistic", "excited", "eager", "looking forward",
                "enthusiastic"],
    "desperate": ["desperate", "hopeless", "lost", "trapped", "stuck",
                  "no choice", "last resort"],
}


def _try_spacy_parse(text: str) -> SituationContext | None:
    try:
        import spacy
    except Exception:
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return None

    doc = nlp(text)
    ctx = SituationContext(raw_input=text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ctx.actors.append(ent.text)
        elif ent.label_ in ("GPE", "LOC", "FAC", "ORG"):
            ctx.environment = ent.text

    for token in doc:
        if token.dep_ == "nsubj" and token.pos_ in ("NOUN", "PROPN", "PRON"):
            if not ctx.main_actor:
                ctx.main_actor = token.text
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            ctx.action_verb = token.lemma_
        if token.dep_ in ("dobj", "attr", "oprd") and token.head.dep_ == "ROOT":
            ctx.object_of_action = token.text

    return ctx


def _fallback_parse(text: str) -> SituationContext:
    ctx = SituationContext(raw_input=text)
    words = text.lower().split()

    pronoun_map = {"i": "self", "my": "self", "me": "self",
                   "we": "self", "he": "other", "she": "other", "they": "others"}
    for word in words:
        if word in pronoun_map and not ctx.main_actor:
            ctx.main_actor = pronoun_map[word]

    if not ctx.main_actor:
        ctx.main_actor = "self"

    return ctx


def _detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, float] = {}

    # action keywords carry more weight -- they indicate what the person is doing
    # context keywords indicate who or what is involved, weighted lower
    ACTION_KEYWORDS: dict[str, list[str]] = {
        "career": ["quit", "resign", "hire", "fire", "promote", "interview",
                    "apply", "startup", "launch"],
        "relationship": ["break up", "divorce", "propose", "confess",
                         "confront", "forgive"],
        "health": ["surgery", "diagnos", "treatment", "therapy", "exercise"],
        "finance": ["invest", "borrow", "loan", "mortgage", "gamble", "trade"],
        "education": ["enroll", "study", "drop out", "graduate", "thesis"],
        "travel": ["move", "relocate", "emigrate", "migrate", "live abroad",
                   "go abroad", "move abroad", "leave for", "move to"],
    }

    for domain, keywords in DOMAIN_KEYWORDS.items():
        context_score = sum(1 for kw in keywords if kw in text_lower)
        action_score = sum(2.5 for kw in ACTION_KEYWORDS.get(domain, [])
                          if kw in text_lower)
        total = context_score + action_score
        if total > 0:
            scores[domain] = total

    if scores:
        return max(scores, key=scores.get)
    return "general"


def _detect_emotion(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[emotion] = score
    if scores:
        return max(scores, key=scores.get)
    return "neutral"


def _extract_stakes(text: str, domain: str) -> list[str]:
    stakes_map = {
        "career": ["professional reputation", "income stability", "career trajectory"],
        "relationship": ["emotional wellbeing", "trust", "long-term companionship"],
        "health": ["physical wellbeing", "quality of life", "longevity"],
        "finance": ["financial security", "wealth accumulation", "debt level"],
        "education": ["knowledge growth", "credentials", "future opportunities"],
        "travel": ["lifestyle change", "social connections", "cultural experience"],
        "general": ["personal growth", "daily routine", "future options"],
    }
    return stakes_map.get(domain, stakes_map["general"])


ROLE_KEYWORDS: dict[str, tuple[str, PowerDynamic]] = {
    "boss": ("manager", PowerDynamic.SUPERIOR),
    "manager": ("manager", PowerDynamic.SUPERIOR),
    "supervisor": ("supervisor", PowerDynamic.SUPERIOR),
    "director": ("director", PowerDynamic.SUPERIOR),
    "ceo": ("executive", PowerDynamic.SUPERIOR),
    "teacher": ("educator", PowerDynamic.SUPERIOR),
    "professor": ("educator", PowerDynamic.SUPERIOR),
    "doctor": ("medical", PowerDynamic.SUPERIOR),
    "judge": ("legal authority", PowerDynamic.SUPERIOR),
    "parent": ("family authority", PowerDynamic.SUPERIOR),
    "mother": ("family authority", PowerDynamic.SUPERIOR),
    "father": ("family authority", PowerDynamic.SUPERIOR),
    "colleague": ("peer", PowerDynamic.EQUAL),
    "coworker": ("peer", PowerDynamic.EQUAL),
    "friend": ("peer", PowerDynamic.EQUAL),
    "classmate": ("peer", PowerDynamic.EQUAL),
    "sibling": ("family peer", PowerDynamic.EQUAL),
    "brother": ("family peer", PowerDynamic.EQUAL),
    "sister": ("family peer", PowerDynamic.EQUAL),
    "partner": ("romantic", PowerDynamic.EQUAL),
    "spouse": ("romantic", PowerDynamic.EQUAL),
    "husband": ("romantic", PowerDynamic.EQUAL),
    "wife": ("romantic", PowerDynamic.EQUAL),
    "girlfriend": ("romantic", PowerDynamic.EQUAL),
    "boyfriend": ("romantic", PowerDynamic.EQUAL),
    "employee": ("subordinate", PowerDynamic.SUBORDINATE),
    "intern": ("subordinate", PowerDynamic.SUBORDINATE),
    "student": ("learner", PowerDynamic.SUBORDINATE),
    "child": ("family dependent", PowerDynamic.SUBORDINATE),
    "son": ("family dependent", PowerDynamic.SUBORDINATE),
    "daughter": ("family dependent", PowerDynamic.SUBORDINATE),
}

PRECONDITION_MARKERS = [
    "but", "however", "although", "even though", "despite",
    "because", "since", "after", "now that", "given that",
    "while", "whereas", "considering",
]

NEGATIVE_PRECONDITION_KEYWORDS = [
    "found out", "caught", "lied", "cheated", "failed", "lost",
    "fired", "rejected", "refused", "argued", "fought", "broke",
    "betrayed", "forgot", "missed", "ignored", "angry", "upset",
]

POSITIVE_PRECONDITION_KEYWORDS = [
    "promoted", "succeeded", "won", "earned", "saved", "helped",
    "praised", "rewarded", "agreed", "supported", "approved",
    "impressed", "trusted", "invited",
]

CONSTRAINT_KEYWORDS = [
    "deadline", "only chance", "last chance", "running out of time",
    "no money", "can't afford", "no other option", "must decide",
    "today", "tomorrow", "tonight", "right now", "immediately",
    "before it's too late", "one shot",
]

TIME_PRESSURE_KEYWORDS = [
    "right now", "immediately", "today", "tonight", "tomorrow",
    "deadline", "last chance", "running out", "hurry", "urgent",
    "soon", "quickly", "before",
]

IRREVERSIBLE_KEYWORDS = [
    "quit", "resign", "divorce", "break up", "drop out", "move",
    "surgery", "sell", "burn bridges", "confess", "propose",
    "marry", "emigrate", "terminate", "abort", "permanent",
    "cannot be undone", "irreversible", "no going back",
    "cannot undo", "point of no return", "final",
]

CONFLICT_KEYWORDS = [
    "disagree", "argument", "conflict", "tension", "against",
    "oppose", "refuse", "deny", "compete", "rival", "jealous",
    "resent", "distrust", "suspicious", "betray", "but i",
    "however", "whereas", "while i", "wants to",
]

VOLATILITY_KEYWORDS: dict[str, float] = {
    "career": 0.45,
    "relationship": 0.55,
    "health": 0.40,
    "finance": 0.65,
    "education": 0.30,
    "travel": 0.50,
    "general": 0.45,
}


def _extract_actors(text: str) -> list[ActorProfile]:
    text_lower = text.lower()
    profiles = []
    seen_roles = set()

    for keyword, (role, power) in ROLE_KEYWORDS.items():
        if keyword in text_lower and role not in seen_roles:
            profiles.append(ActorProfile(
                name=keyword,
                role=role,
                relationship_to_main=role,
                power_dynamic=power,
            ))
            seen_roles.add(role)

    return profiles


def _extract_preconditions(text: str) -> list[Precondition]:
    preconditions = []
    text_lower = text.lower()

    for marker in PRECONDITION_MARKERS:
        pattern = re.compile(rf"\b{re.escape(marker)}\b(.+?)(?:[.,;]|$)", re.IGNORECASE)
        matches = pattern.findall(text)
        for match in matches:
            clause = match.strip()
            if len(clause) < 3:
                continue

            sentiment = 0.0
            trust_mod = 0.0
            urgency_mod = 0.0

            clause_lower = clause.lower()
            for neg_kw in NEGATIVE_PRECONDITION_KEYWORDS:
                if neg_kw in clause_lower:
                    sentiment -= 0.2
                    trust_mod -= 0.15

            for pos_kw in POSITIVE_PRECONDITION_KEYWORDS:
                if pos_kw in clause_lower:
                    sentiment += 0.2
                    trust_mod += 0.1

            for time_kw in TIME_PRESSURE_KEYWORDS:
                if time_kw in clause_lower:
                    urgency_mod += 0.15

            preconditions.append(Precondition(
                description=clause.strip(),
                sentiment_impact=max(-1.0, min(1.0, sentiment)),
                trust_modifier=max(-1.0, min(1.0, trust_mod)),
                urgency_modifier=max(0.0, min(1.0, urgency_mod)),
            ))

    return preconditions


def _extract_constraints(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for kw in CONSTRAINT_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return found


def _detect_time_pressure(text: str) -> float:
    text_lower = text.lower()
    score = 0.0
    for kw in TIME_PRESSURE_KEYWORDS:
        if kw in text_lower:
            score += 0.2
    return min(1.0, score)


def _detect_reversibility(text: str) -> Reversibility:
    text_lower = text.lower()
    irreversible_count = sum(1 for kw in IRREVERSIBLE_KEYWORDS if kw in text_lower)
    if irreversible_count >= 2:
        return Reversibility.IRREVERSIBLE
    if irreversible_count == 1:
        return Reversibility.DIFFICULT
    return Reversibility.MODERATE


def _detect_conflicts(text: str, actors: list[ActorProfile]) -> list[ConflictVector]:
    text_lower = text.lower()
    conflicts = []

    has_conflict_language = any(kw in text_lower for kw in CONFLICT_KEYWORDS)
    if not has_conflict_language:
        return conflicts

    for actor in actors:
        if actor.power_dynamic != PowerDynamic.UNKNOWN:
            conflicts.append(ConflictVector(
                actor_a="self",
                actor_b=actor.name,
                nature=f"power-asymmetric conflict with {actor.role}",
                intensity=0.6 if actor.is_authority else 0.4,
            ))

    if not conflicts and has_conflict_language:
        conflicts.append(ConflictVector(
            actor_a="self",
            actor_b="other",
            nature="interpersonal conflict",
            intensity=0.5,
        ))

    return conflicts


def _compute_stake_severity(domain: str, text: str,
                            preconditions: list[Precondition],
                            reversibility: Reversibility) -> float:
    base = {"career": 0.6, "relationship": 0.6, "health": 0.7, "finance": 0.65,
            "education": 0.4, "travel": 0.45, "general": 0.4}.get(domain, 0.4)

    precondition_shift = sum(abs(p.sentiment_impact) for p in preconditions) * 0.1

    reversibility_shift = {
        Reversibility.IRREVERSIBLE: 0.25,
        Reversibility.DIFFICULT: 0.15,
        Reversibility.MODERATE: 0.0,
        Reversibility.EASY: -0.1,
    }.get(reversibility, 0.0)

    return min(1.0, max(0.1, base + precondition_shift + reversibility_shift))


def _try_ml_parse(text: str) -> dict | None:
    try:
        from simulus.ml.ml_parser import is_model_available, predict
        if not is_model_available():
            return None
        return predict(text)
    except Exception:
        return None


ML_CONFIDENCE_THRESHOLD = 0.5


def parse_situation(text: str) -> SituationContext:
    ctx = _try_spacy_parse(text)
    if ctx is None:
        ctx = _fallback_parse(text)

    # try ML model first, fall back to keyword heuristics
    ml_result = _try_ml_parse(text)
    keyword_domain = _detect_domain(text)
    keyword_emotion = _detect_emotion(text)

    if ml_result is not None:
        ml_domain = ml_result["domain"]
        ml_emotion = ml_result["emotion"]
        domain_conf = ml_result["domain_confidence"]
        emotion_conf = ml_result["emotion_confidence"]
        ctx.domain = ml_domain if domain_conf >= ML_CONFIDENCE_THRESHOLD else keyword_domain
        ctx.emotional_state = ml_emotion if emotion_conf >= ML_CONFIDENCE_THRESHOLD else keyword_emotion
        ctx.ml_signals = {
            "domain_distribution": ml_result.get("domain_distribution", {}),
            "emotion_distribution": ml_result.get("emotion_distribution", {}),
            "domain_confidence": domain_conf,
            "emotion_confidence": emotion_conf,
        }
    else:
        ctx.domain = keyword_domain
        ctx.emotional_state = keyword_emotion

    ctx.stakes = _extract_stakes(text, ctx.domain)

    ctx.actor_profiles = _extract_actors(text)
    if ctx.actor_profiles:
        for profile in ctx.actor_profiles:
            if profile.name not in ctx.actors:
                ctx.actors.append(profile.name)

    ctx.preconditions = _extract_preconditions(text)
    ctx.constraints = _extract_constraints(text)
    ctx.time_pressure = _detect_time_pressure(text)
    ctx.reversibility = _detect_reversibility(text)
    ctx.conflict_vectors = _detect_conflicts(text, ctx.actor_profiles)

    ctx.stake_severity = _compute_stake_severity(
        ctx.domain, text, ctx.preconditions, ctx.reversibility)

    ctx.volatility = VOLATILITY_KEYWORDS.get(ctx.domain, 0.45)

    if not ctx.main_actor:
        ctx.main_actor = "self"

    return ctx
