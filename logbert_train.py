import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from torch.nn import CrossEntropyLoss
import numpy as np

# Load BATADAL logs with labels
df = pd.read_csv("/Users/nidhikulkarni/ics-anomaly-detection/data/BATADAL/train_dataset.csv")

logs = []
labels = []
for _, row in df.iterrows():
    # same log format as before
    log_line = f"[TIME={row['DATETIME']:.4f}] "
    for col in df.columns[2:-1]:  # skip ID, DATETIME, keep sensors & status
        log_line += f"{col}={row[col]:.2f} "
    logs.append(log_line.strip())
    labels.append(row["ATT_FLAG"])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class LogDataset(Dataset):
    def __init__(self, logs, labels, tokenizer, max_len=128):
        self.logs = logs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        text = self.logs[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten(),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }

dataset = LogDataset(logs, labels, tokenizer)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./logbert_model",
    #evaluation_strategy="epoch",  # works in new versions


    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save final model
model.save_pretrained("./logbert/final")
tokenizer.save_pretrained("./logbert/final")



def anomaly_score(logs, model, tokenizer, max_len=128):
    model.eval()
    scores = []
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="mean")

    for text in logs:
        encoding = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
        with torch.no_grad():
            outputs = model(**encoding, labels=encoding["input_ids"])
            loss = outputs.loss.item()
        scores.append(loss)
    return np.array(scores)

# Compute anomaly scores
scores = anomaly_score(logs, model, tokenizer)

# Evaluate against ATT_FLAG
y_true = np.array(labels)
auc = roc_auc_score(y_true, scores)

# Binarize using threshold = median (you can tune this)
threshold = np.median(scores)
y_pred = (scores > threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

print("🔹 Evaluation Results")
print(f"AUC-ROC: {auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
