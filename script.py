"""
script.py
====================
CAIL2018 罪名多標籤預測｜ELECTRA-base 全參數 fine-tuning
模型：hfl/chinese-legal-electra-base-discriminator
結果：Test Micro F1 = 0.8318、Test Macro F1 = 0.7476（8 epoch）

使用方式：
    python script.py
路徑設定在下方 class Args，依實際位置修改。
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ══════════════════════════════════════════════════════════════════════════════
# 設定區（路徑與超參數全部集中在這裡）
# ══════════════════════════════════════════════════════════════════════════════
class Args:
    train_path  = r"CAIL2018_ALL_DATA\final_all_data\data_train.json"
    valid_path  = r"CAIL2018_ALL_DATA\final_all_data\data_valid.json"
    test_path   = r"CAIL2018_ALL_DATA\final_all_data\data_test.json"
    model_name  = "hfl/chinese-legal-electra-base-discriminator"
    output_dir  = r"saved_model_v4"
    plot_path   = r"saved_model_v4\training_curve.png"

    max_len            = 512
    batch_size         = 2
    accumulation_steps = 8      # 等效 batch size = 2 × 8 = 16
    lr                 = 1e-4
    dropout_rate       = 0.2
    threshold          = 0.5
    epochs             = 8
    warmup_ratio       = 0.1
    seed               = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1. 資料讀取
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path):
    facts, accusations = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            facts.append(item.get("fact", ""))
            accusations.append(item.get("meta", {}).get("accusation", []))
    return facts, accusations


# ══════════════════════════════════════════════════════════════════════════════
# 2. pos_weight（log1p 平滑）
# ══════════════════════════════════════════════════════════════════════════════
def compute_pos_weight(labels):
    total  = labels.shape[0]
    counts = labels.sum(axis=0)
    counts = np.where(counts == 0, 1, counts)
    return torch.tensor(np.log1p(total / counts), dtype=torch.float)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class CailDataset(Dataset):
    def __init__(self, facts, labels, tokenizer, max_len):
        self.facts     = facts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.facts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. 模型（單層分類頭，CLS pooling）
# ══════════════════════════════════════════════════════════════════════════════
class ElectraClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.2):
        super().__init__()
        self.electra = AutoModel.from_pretrained(model_name)
        hidden_size  = self.electra.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        cls     = outputs.last_hidden_state[:, 0, :]   # CLS token
        return self.classifier(cls)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 訓練一個 epoch
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = loss_fn(logits, labels) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % (500 * accum_steps) == 0:
            print(f"  Step {step+1}/{len(loader)}, Loss: {loss.item() * accum_steps:.4f}",
                  flush=True)

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 評估
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            logits     = model(input_ids, attention_mask)
            total_loss += loss_fn(logits, labels).item()
            preds       = (torch.sigmoid(logits) > threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return total_loss / len(loader), {
        "micro_p":     precision_score(all_labels, all_preds, average="micro",    zero_division=0),
        "micro_r":     recall_score   (all_labels, all_preds, average="micro",    zero_division=0),
        "micro_f1":    f1_score       (all_labels, all_preds, average="micro",    zero_division=0),
        "macro_p":     precision_score(all_labels, all_preds, average="macro",    zero_division=0),
        "macro_r":     recall_score   (all_labels, all_preds, average="macro",    zero_division=0),
        "macro_f1":    f1_score       (all_labels, all_preds, average="macro",    zero_division=0),
        "weighted_f1": f1_score       (all_labels, all_preds, average="weighted", zero_division=0),
        "samples_f1":  f1_score       (all_labels, all_preds, average="samples",  zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. 視覺化
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history, save_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ELECTRA-base Fine-tuning（CAIL2018 罪名預測）", fontsize=14)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["valid_loss"], marker="o", label="Valid Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["micro_f1"],    marker="o", label="Micro F1")
    axes[1].plot(epochs, history["macro_f1"],    marker="o", label="Macro F1")
    axes[1].plot(epochs, history["weighted_f1"], marker="s", label="Weighted F1", linestyle="--")
    axes[1].plot(epochs, history["samples_f1"],  marker="s", label="Samples F1",  linestyle="--")
    axes[1].set_title("F1 Scores")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"訓練曲線已儲存：{save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8. 主程式
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args   = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 資料載入
    print("讀取資料...")
    train_facts, train_accus = load_data(args.train_path)
    valid_facts, valid_accus = load_data(args.valid_path)
    test_facts,  test_accus  = load_data(args.test_path)
    print(f"訓練集：{len(train_facts)} 筆 | 驗證集：{len(valid_facts)} 筆 | 測試集：{len(test_facts)} 筆")

    # 多標籤二值化
    mlb          = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_accus)
    valid_labels = mlb.transform(valid_accus)
    test_labels  = mlb.transform(test_accus)
    num_labels   = len(mlb.classes_)
    print(f"罪名類別數：{num_labels}")

    # pos_weight
    pos_weight = compute_pos_weight(train_labels).to(device)

    # Tokenizer & DataLoader
    tokenizer    = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(
        CailDataset(train_facts, train_labels, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=True,
    )
    valid_loader = DataLoader(
        CailDataset(valid_facts, valid_labels, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader  = DataLoader(
        CailDataset(test_facts, test_labels, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=False,
    )

    # 模型
    print("載入模型...")
    model   = ElectraClassifier(args.model_name, num_labels, args.dropout_rate).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps   = (len(train_loader) // args.accumulation_steps) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 訓練迴圈
    best_micro_f1 = 0.0
    history = {k: [] for k in [
        "train_loss", "valid_loss",
        "micro_p", "micro_r", "micro_f1",
        "macro_p", "macro_r", "macro_f1",
        "weighted_f1", "samples_f1",
    ]}

    print(f"\n開始訓練（{args.epochs} epochs）...\n")

    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        train_loss          = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, args.accumulation_steps,
        )
        val_loss, metrics   = evaluate(model, valid_loader, loss_fn, device, args.threshold)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Valid Loss : {val_loss:.4f}")
        print(f"  Micro  P : {metrics['micro_p']:.4f}  |  Micro  R : {metrics['micro_r']:.4f}"
              f"  |  Micro  F1 : {metrics['micro_f1']:.4f}")
        print(f"  Macro  P : {metrics['macro_p']:.4f}  |  Macro  R : {metrics['macro_r']:.4f}"
              f"  |  Macro  F1 : {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1 : {metrics['weighted_f1']:.4f}"
              f"  |  Samples F1 : {metrics['samples_f1']:.4f}\n")

        # 記錄歷史
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(val_loss)
        for k in metrics:
            history[k].append(metrics[k])

        # 儲存最佳模型（以 valid Micro F1 為基準）
        if metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = metrics["micro_f1"]
            model.electra.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(
                model.classifier.state_dict(),
                os.path.join(args.output_dir, "classifier_head.pt"),
            )
            print(f"✓ Best model 更新（Valid Micro F1 = {best_micro_f1:.4f}）\n")

    # 訓練曲線
    plot_training_curves(history, args.plot_path)

    # ── Test Set 評估 ─────────────────────────────────────────────────────
    print("載入 best model，評估 Test Set...")
    model.electra = AutoModel.from_pretrained(args.output_dir)
    model.classifier.load_state_dict(
        torch.load(
            os.path.join(args.output_dir, "classifier_head.pt"),
            map_location=device,
        )
    )
    model = model.to(device)

    _, test_metrics = evaluate(model, test_loader, loss_fn, device, args.threshold)

    print("\n" + "=" * 60)
    print("Test Set 最終結果（可寫入報告）")
    print("=" * 60)
    print(f"  Micro  P : {test_metrics['micro_p']:.4f}"
          f"  |  Micro  R : {test_metrics['micro_r']:.4f}"
          f"  |  Micro  F1 : {test_metrics['micro_f1']:.4f}")
    print(f"  Macro  P : {test_metrics['macro_p']:.4f}"
          f"  |  Macro  R : {test_metrics['macro_r']:.4f}"
          f"  |  Macro  F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1 : {test_metrics['weighted_f1']:.4f}"
          f"  |  Samples F1 : {test_metrics['samples_f1']:.4f}")
    print("=" * 60)
    print(f"\n模型儲存於：{args.output_dir}")


if __name__ == "__main__":
    main()