from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import DistilBertTokenizer, DistilBertModel

from simulus.ml.train import SituationClassifier, DOMAINS, EMOTIONS, MODEL_DIR


_cached_model: SituationClassifier | None = None
_cached_tokenizer: DistilBertTokenizer | None = None
_cached_device: torch.device | None = None


def is_model_available() -> bool:
    return (MODEL_DIR / "classifier.pt").exists()


def _load_model() -> tuple[SituationClassifier, DistilBertTokenizer, torch.device]:
    global _cached_model, _cached_tokenizer, _cached_device

    if _cached_model is not None:
        return _cached_model, _cached_tokenizer, _cached_device

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))

    model = SituationClassifier().to(device)

    state = torch.load(MODEL_DIR / "classifier.pt", map_location=device, weights_only=True)
    model.shared.load_state_dict(state["shared"])
    model.domain_head.load_state_dict(state["domain_head"])
    model.emotion_head.load_state_dict(state["emotion_head"])
    model.bert.transformer.layer[-2:].load_state_dict(state["bert_layers"])

    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_device = device

    return model, tokenizer, device


def predict(text: str) -> dict:
    model, tokenizer, device = _load_model()

    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        domain_logits, emotion_logits = model(input_ids, attention_mask)

    domain_probs = torch.softmax(domain_logits, dim=1).squeeze(0).cpu().numpy()
    emotion_probs = torch.softmax(emotion_logits, dim=1).squeeze(0).cpu().numpy()

    domain_idx = domain_probs.argmax()
    emotion_idx = emotion_probs.argmax()

    return {
        "domain": DOMAINS[domain_idx],
        "domain_confidence": float(domain_probs[domain_idx]),
        "domain_distribution": {d: float(p) for d, p in zip(DOMAINS, domain_probs)},
        "emotion": EMOTIONS[emotion_idx],
        "emotion_confidence": float(emotion_probs[emotion_idx]),
        "emotion_distribution": {e: float(p) for e, p in zip(EMOTIONS, emotion_probs)},
    }
