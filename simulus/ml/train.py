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
                 hidden_dim: int = 256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # freeze lower layers, only fine-tune top 2 transformer blocks
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.transformer.layer[-2:].parameters():
            param.requires_grad = True

        bert_dim = self.bert.config.hidden_size

        self.shared = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.domain_head = nn.Linear(hidden_dim, n_domains)
        self.emotion_head = nn.Linear(hidden_dim, n_emotions)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        shared = self.shared(cls_output)
        domain_logits = self.domain_head(shared)
        emotion_logits = self.emotion_head(shared)
        return domain_logits, emotion_logits


def train(data_path: str = "data/training_data.json",
          epochs: int = 8,
          batch_size: int = 16,
          lr: float = 2e-5,
          val_split: float = 0.15) -> dict:

    with open(data_path) as f:
        data = json.load(f)

    # split
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
    )
    domain_criterion = nn.CrossEntropyLoss()
    emotion_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
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
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validation
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

        results["train_loss"].append(avg_loss)
        results["val_domain_acc"].append(domain_acc)
        results["val_emotion_acc"].append(emotion_acc)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Domain Acc: {domain_acc:.3f} | Emotion Acc: {emotion_acc:.3f}")

        if domain_acc + emotion_acc > best_val_acc:
            best_val_acc = domain_acc + emotion_acc
            _save_model(model, tokenizer)
            print(f"  -> Saved best model (combined acc: {best_val_acc:.3f})")

    return results


def _save_model(model: SituationClassifier,
                tokenizer: DistilBertTokenizer) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # save only the classifier heads and shared layer (not the full bert)
    state = {
        "shared": model.shared.state_dict(),
        "domain_head": model.domain_head.state_dict(),
        "emotion_head": model.emotion_head.state_dict(),
        "bert_layers": model.bert.transformer.layer[-2:].state_dict(),
    }
    torch.save(state, MODEL_DIR / "classifier.pt")
    tokenizer.save_pretrained(str(MODEL_DIR / "tokenizer"))

    meta = {"domains": DOMAINS, "emotions": EMOTIONS}
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    from simulus.ml.generate_data import generate_dataset

    print("Generating training data...")
    generate_dataset(n_samples=3000, output_path="data/training_data.json")

    print("Training classifier...")
    results = train("data/training_data.json", epochs=15)

    print(f"\nFinal domain accuracy: {results['val_domain_acc'][-1]:.3f}")
    print(f"Final emotion accuracy: {results['val_emotion_acc'][-1]:.3f}")
