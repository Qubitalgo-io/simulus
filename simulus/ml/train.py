from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel


DOMAINS = ["career", "relationship", "health", "finance", "education", "travel"]
EMOTIONS = ["anxious", "confident", "angry", "hopeful", "desperate", "neutral"]

MODEL_DIR = Path(__file__).parent / "model"

N_UNFROZEN_LAYERS = 4


class SituationDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: DistilBertTokenizer,
                 max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
        self.emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "domain_label": torch.tensor(self.domain_to_idx[item["domain"]]),
            "emotion_label": torch.tensor(self.emotion_to_idx[item["emotion"]]),
        }


class SituationClassifier(nn.Module):
    def __init__(self, n_domains: int = len(DOMAINS),
                 n_emotions: int = len(EMOTIONS),
                 hidden_dim: int = 384,
                 n_unfrozen: int = N_UNFROZEN_LAYERS):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.n_unfrozen = n_unfrozen

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer.layer[-n_unfrozen:].parameters():
            param.requires_grad = True

        bert_dim = self.bert.config.hidden_size

        self.shared = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, n_domains),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, n_emotions),
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        shared = self.shared(cls_output)
        domain_logits = self.domain_head(shared)
        emotion_logits = self.emotion_head(shared)
        return domain_logits, emotion_logits


def _compute_class_weights(data: list[dict], key: str,
                           labels: list[str]) -> torch.Tensor:
    counts = {label: 0 for label in labels}
    for item in data:
        counts[item[key]] += 1
    total = len(data)
    weights = []
    for label in labels:
        c = max(counts[label], 1)
        weights.append(total / (len(labels) * c))
    return torch.tensor(weights, dtype=torch.float32)


def train(data_path: str = "data/training_data.json",
          epochs: int = 20,
          batch_size: int = 16,
          lr: float = 3e-5,
          val_split: float = 0.15,
          patience: int = 5,
          label_smoothing: float = 0.1) -> dict:

    with open(data_path) as f:
        data = json.load(f)

    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * (1 - val_split))
    train_data = [data[i] for i in indices[:split_idx]]
    val_data = [data[i] for i in indices[split_idx:]]

    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = SituationDataset(train_data, tokenizer)
    val_dataset = SituationDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SituationClassifier().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6,
    )

    domain_weights = _compute_class_weights(train_data, "domain", DOMAINS).to(device)
    emotion_weights = _compute_class_weights(train_data, "emotion", EMOTIONS).to(device)

    domain_criterion = nn.CrossEntropyLoss(
        weight=domain_weights, label_smoothing=label_smoothing,
    )
    emotion_criterion = nn.CrossEntropyLoss(
        weight=emotion_weights, label_smoothing=label_smoothing,
    )

    best_val_acc = 0.0
    patience_counter = 0
    results = {"train_loss": [], "val_domain_acc": [], "val_emotion_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            domain_labels = batch["domain_label"].to(device)
            emotion_labels = batch["emotion_label"].to(device)

            domain_logits, emotion_logits = model(input_ids, attention_mask)
            loss = domain_criterion(domain_logits, domain_labels) + \
                   emotion_criterion(emotion_logits, emotion_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        domain_correct = 0
        emotion_correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                domain_labels = batch["domain_label"].to(device)
                emotion_labels = batch["emotion_label"].to(device)

                domain_logits, emotion_logits = model(input_ids, attention_mask)
                domain_preds = domain_logits.argmax(dim=1)
                emotion_preds = emotion_logits.argmax(dim=1)

                domain_correct += (domain_preds == domain_labels).sum().item()
                emotion_correct += (emotion_preds == emotion_labels).sum().item()
                total += len(domain_labels)

        domain_acc = domain_correct / total
        emotion_acc = emotion_correct / total
        combined = domain_acc + emotion_acc

        results["train_loss"].append(avg_loss)
        results["val_domain_acc"].append(domain_acc)
        results["val_emotion_acc"].append(emotion_acc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Domain: {domain_acc:.3f} | Emotion: {emotion_acc:.3f} | "
              f"LR: {current_lr:.2e}")

        if combined > best_val_acc:
            best_val_acc = combined
            patience_counter = 0
            _save_model(model, tokenizer)
            print(f"  -> Saved best model (combined acc: {best_val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break

    return results


def _save_model(model: SituationClassifier,
                tokenizer: DistilBertTokenizer) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "shared": model.shared.state_dict(),
        "domain_head": model.domain_head.state_dict(),
        "emotion_head": model.emotion_head.state_dict(),
        "bert_layers": model.bert.transformer.layer[-model.n_unfrozen:].state_dict(),
        "n_unfrozen": model.n_unfrozen,
    }
    torch.save(state, MODEL_DIR / "classifier.pt")
    tokenizer.save_pretrained(str(MODEL_DIR / "tokenizer"))

    meta = {"domains": DOMAINS, "emotions": EMOTIONS,
            "n_unfrozen": model.n_unfrozen}
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    from simulus.ml.generate_data import generate_dataset

    print("Generating training data...")
    generate_dataset(n_samples=5000, output_path="data/training_data.json")

    print("Training classifier...")
    results = train("data/training_data.json", epochs=20)

    print(f"\nFinal domain accuracy: {results['val_domain_acc'][-1]:.3f}")
    print(f"Final emotion accuracy: {results['val_emotion_acc'][-1]:.3f}")
