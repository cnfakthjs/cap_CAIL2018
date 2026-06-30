"""
script_compare.py
================================
模型   : hfl/chinese-legal-electra-base-discriminator
設定   : 完全照組員腳本（attention pooling, 兩層分類頭, pos_weight cap=10,
         dropout=0.1, batch=4, encoder_lr=2e-5, head_lr=1e-3, warmup=10%）
差異   : 1) ELECTRA-base（非 small）
         2) FREEZE_ENCODER = False（full finetune，全解凍）
         3) 篩選邏輯：fact字數 > 55（嚴格大於）、罪名頻率 > 50（嚴格大於）→ 160 類
輸出   : saved_model_v4/
           best_model.pt      ← valid macro F1 最高的 checkpoint
           last_checkpoint.pt ← 最後一個 epoch 的 checkpoint
           label_map.pt       ← label2id / id2label 對照表
           test_results.json  ← test set 最終指標

使用方式（cd 進 electra 資料夾後）：
    python script_compare.py
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ══════════════════════════════════════════════════════════════════════════════
# 超參數（照組員 config.py，僅標注與組員不同之處）
# ══════════════════════════════════════════════════════════════════════════════
class Args:
    # 資料路徑
    train      = r"CAIL2018_ALL_DATA\no_repeat_data\data_train.json"
    valid      = r"CAIL2018_ALL_DATA\no_repeat_data\data_valid.json"
    test       = r"CAIL2018_ALL_DATA\no_repeat_data\data_test.json"
    output_dir = r"saved_model_v4"

    # 模型（與組員不同：base 而非 small）
    pretrained = "hfl/chinese-legal-electra-base-discriminator"

    # 訓練設定（照組員 config.py）
    max_len       = 512
    batch_size    = 2      # 實體 batch（省 VRAM）；accumulation_steps=2 → 等效 batch 4
    accumulation_steps = 2  # 每 2 步更新一次，等效 batch size = 2 × 2 = 4（對齊組員）
    epochs        = 1
    encoder_lr    = 2e-5
    head_lr       = 1e-3
    warmup_ratio  = 0.1
    weight_decay  = 0.01
    dropout       = 0.1
    fc_hidden     = 512
    pooling_mode  = "attention"
    threshold     = 0.5
    seed          = 42
    pos_weight_cap = 10.0

    # 篩選邏輯（嚴格大於，對齊組員實際用的條件）
    min_fact_chars = 55   # fact 字數 > 55
    min_label_freq = 50   # 罪名頻率 > 50（嚴格大於，非 >=）

    # 凍結設定（與組員不同：全解凍）
    freeze_encoder = False


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_label(s: str) -> str:
    return s.replace("[", "").replace("]", "").strip()


def iter_accusations(records: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in records:
        acc = r.get("meta", {}).get("accusation", [])
        if isinstance(acc, str):
            out.append(normalize_label(acc))
        else:
            for a in acc:
                out.append(normalize_label(str(a)))
    return out


def filter_by_fact_length(
    records: List[Dict[str, Any]], min_chars: int
) -> List[Dict[str, Any]]:
    """保留 fact 字數 > min_chars（嚴格大於）"""
    return [r for r in records if len(r.get("fact", "")) > min_chars]


def build_label_maps(
    train_records: List[Dict[str, Any]], min_freq: int
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    篩選罪名：出現次數 > min_freq（嚴格大於）
    注意：組員原版是 >= ，這裡改成 > 以對齊實際產生 160 類的條件
    """
    counts = Counter(iter_accusations(train_records))
    kept = sorted([lab for lab, c in counts.items() if c > min_freq])
    label2id = {lab: i for i, lab in enumerate(kept)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def record_to_multihot(
    record: Dict[str, Any], label2id: Dict[str, int]
) -> Optional[np.ndarray]:
    acc = record.get("meta", {}).get("accusation", [])
    if isinstance(acc, str):
        names = [normalize_label(acc)]
    else:
        names = [normalize_label(str(a)) for a in acc]
    vec = np.zeros(len(label2id), dtype=np.float32)
    any_kept = False
    for name in names:
        if name in label2id:
            vec[label2id[name]] = 1.0
            any_kept = True
    return vec if any_kept else None


def build_dataset_arrays(
    records: List[Dict[str, Any]], label2id: Dict[str, int]
) -> Tuple[List[str], List[np.ndarray], List[bool]]:
    facts, labels, has_label = [], [], []
    for r in records:
        fact = r.get("fact", "")
        acc = r.get("meta", {}).get("accusation")
        if acc:
            vec = record_to_multihot(r, label2id)
            if vec is None:
                continue
            facts.append(fact)
            labels.append(vec)
            has_label.append(True)
        else:
            facts.append(fact)
            labels.append(np.zeros(len(label2id), dtype=np.float32))
            has_label.append(False)
    return facts, labels, has_label


def compute_pos_weights(
    labels: List[np.ndarray], cap: float
) -> torch.Tensor:
    mat = np.stack(labels, axis=0)
    pos_c = mat.sum(axis=0).astype(np.float64)
    n = float(mat.shape[0])
    weights = (n - pos_c) / np.maximum(pos_c, 1.0)
    weights = np.minimum(weights, cap)
    return torch.from_numpy(weights.astype(np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# Dataset（照組員 dataset.py）
# ══════════════════════════════════════════════════════════════════════════════
class LegalChargeDataset(Dataset):
    def __init__(
        self,
        facts: List[str],
        labels: List[np.ndarray],
        tokenizer_name: str,
        max_length: int,
        has_label: Optional[List[bool]] = None,
    ):
        self.facts = facts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.has_label = has_label if has_label is not None else [True] * len(facts)

    def __len__(self) -> int:
        return len(self.facts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.facts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.from_numpy(self.labels[idx]),
            "has_label":      torch.tensor(
                                  1.0 if self.has_label[idx] else 0.0,
                                  dtype=torch.float32
                              ),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out = {
        "input_ids":      torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"] for b in batch]),
        "has_label":      torch.stack([b["has_label"] for b in batch]),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 模型（照組員 model.py）
# ══════════════════════════════════════════════════════════════════════════════
class BertAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str,
        dropout: float = 0.1,
        fc_hidden: int = 512,
        pooling_mode: str = "attention",
    ):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        d = int(self.encoder.config.hidden_size)
        self.attn_proj   = nn.Linear(d, d)
        self.attn_scorer = nn.Linear(d, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def attention_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        u = self.attn_scorer(torch.tanh(self.attn_proj(hidden_states))).squeeze(-1)
        u = u.masked_fill(attention_mask == 0, -1e9)
        alpha = torch.softmax(u, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1)

    def mean_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        h = self.encoder(**kwargs).last_hidden_state

        if self.pooling_mode == "cls":
            pooled = h[:, 0]
        elif self.pooling_mode == "mean":
            pooled = self.mean_pooling(h, attention_mask)
        else:  # attention
            pooled = self.attention_pooling(h, attention_mask)

        x = self.dropout(torch.relu(self.fc1(pooled)))
        return self.fc2(x)


# ══════════════════════════════════════════════════════════════════════════════
# 評估
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(
    model: BertAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        sel = batch["has_label"] > 0.5
        if not sel.any():
            continue
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        preds  = (torch.sigmoid(logits[sel]) > threshold).cpu().numpy()
        labels = batch["labels"][sel].numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    if not all_preds:
        return {"macro_f1": 0.0, "micro_f1": 0.0}

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro",  zero_division=0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = Args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"裝置：{device}")
    print(f"輸出目錄：{out_dir}")

    # ── 資料載入 ────────────────────────────────────────────────────────────
    print("載入資料...")
    train_records = load_jsonl(args.train)
    valid_records = load_jsonl(args.valid)
    test_records  = load_jsonl(args.test)

    # fact 字數 > 55（嚴格大於）
    for split, recs in [("train", train_records), ("valid", valid_records), ("test", test_records)]:
        before = len(recs)
        recs = filter_by_fact_length(recs, args.min_fact_chars)
        print(f"  {split}: {before} → {len(recs)} 筆（fact > {args.min_fact_chars} 字）")
        if split == "train":
            train_records = recs
        elif split == "valid":
            valid_records = recs
        else:
            test_records = recs

    # 罪名頻率 > 50（嚴格大於），建立 label 對照表
    label2id, id2label = build_label_maps(train_records, args.min_label_freq)
    num_classes = len(label2id)
    print(f"罪名類別數：{num_classes}（頻率 > {args.min_label_freq}）")

    # 儲存 label map
    torch.save({"label2id": label2id, "id2label": id2label}, out_dir / "label_map.pt")

    # 轉成 multi-hot 矩陣
    tr_facts, tr_labels, tr_has = build_dataset_arrays(train_records, label2id)
    va_facts, va_labels, va_has = build_dataset_arrays(valid_records, label2id)
    te_facts, te_labels, te_has = build_dataset_arrays(test_records,  label2id)

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    train_ds = LegalChargeDataset(tr_facts, tr_labels, args.pretrained, args.max_len, tr_has)
    valid_ds = LegalChargeDataset(va_facts, va_labels, args.pretrained, args.max_len, va_has)
    test_ds  = LegalChargeDataset(te_facts, te_labels, args.pretrained, args.max_len, te_has)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_batch, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_batch, num_workers=0)

    print(f"Train: {len(train_ds)} 筆 / {len(train_loader)} batches")
    print(f"Valid: {len(valid_ds)} 筆  |  Test: {len(test_ds)} 筆")

    # ── 模型 ─────────────────────────────────────────────────────────────────
    print("載入模型...")
    model = BertAttentionClassifier(
        num_classes=num_classes,
        pretrained_model_name=args.pretrained,
        dropout=args.dropout,
        fc_hidden=args.fc_hidden,
        pooling_mode=args.pooling_mode,
    ).to(device)

    # 全解凍（freeze_encoder = False，所有參數都更新）
    encoder_params = list(model.encoder.parameters())
    head_params    = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.")]

    # Gradient checkpointing（照組員 USE_GRADIENT_CHECKPOINTING=True，省 VRAM）
    model.encoder.gradient_checkpointing_enable()
    print("Gradient checkpointing 已啟用")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"可訓練參數：{trainable:,} / 總參數：{total:,}（{trainable/total*100:.1f}%）")

    # ── Loss & Optimizer（differential LR，照組員設定）────────────────────
    pos_weight = compute_pos_weights(tr_labels, args.pos_weight_cap).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": head_params,    "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── 訓練迴圈 ─────────────────────────────────────────────────────────────
    best_macro = -1.0
    best_ckpt  = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            labels         = batch["labels"].to(device)
            has_label      = batch["has_label"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits    = model(input_ids, attention_mask, token_type_ids)
            loss_mat  = criterion(logits, labels)           # [B, C]
            loss_per  = loss_mat.mean(dim=1)                # [B]
            loss      = (loss_per * has_label).sum() / has_label.sum().clamp(min=1.0)
            # gradient accumulation：loss 除以 accumulation_steps 再 backward
            (loss / args.accumulation_steps).backward()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            # 每 accumulation_steps 步才更新一次參數
            if step % args.accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 200 == 0 or step == len(train_loader):
                lrs = scheduler.get_last_lr()
                le  = lrs[0] if lrs else 0.0
                lh  = lrs[1] if len(lrs) > 1 else le
                print(
                    f"  Epoch {epoch} | Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"encoder_lr: {le:.2e} | head_lr: {lh:.2e}",
                    flush=True,
                )

        avg_loss = total_loss / max(n_samples, 1)
        elapsed  = time.perf_counter() - t0

        # Validation
        val_metrics = evaluate(model, valid_loader, device, args.threshold)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Valid Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Valid Micro F1: {val_metrics['micro_f1']:.4f} | "
            f"Time: {elapsed:.0f}s",
            flush=True,
        )

        # Best checkpoint
        if val_metrics["macro_f1"] > best_macro:
            best_macro = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_macro_f1": best_macro,
                    "num_classes": num_classes,
                    "label2id": label2id,
                    "id2label": id2label,
                },
                best_ckpt,
            )
            print(f"  ✓ Best model 更新！Valid Macro F1 = {best_macro:.4f}")

        # Last checkpoint（方便之後 resume）
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_macro_f1": val_metrics["macro_f1"],
                "num_classes": num_classes,
                "label2id": label2id,
                "id2label": id2label,
            },
            out_dir / "last_checkpoint.pt",
        )

    # ── Test Set 評估 ─────────────────────────────────────────────────────────
    print("\n載入 best model，評估 Test Set...")
    blob = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, args.threshold)

    print("=" * 60)
    print("Test Set 最終結果（可寫入報告）")
    print("=" * 60)
    print(f"  Test Macro F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Test Micro F1 : {test_metrics['micro_f1']:.4f}")
    print(f"  Best Valid Macro F1 : {best_macro:.4f}")
    print("=" * 60)

    # 存成 json
    result = {
        "test_macro_f1":       round(test_metrics["macro_f1"], 4),
        "test_micro_f1":       round(test_metrics["micro_f1"], 4),
        "best_valid_macro_f1": round(best_macro, 4),
        "settings": {
            "model":         args.pretrained,
            "freeze_encoder": args.freeze_encoder,
            "pooling_mode":  args.pooling_mode,
            "dropout":       args.dropout,
            "fc_hidden":     args.fc_hidden,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "encoder_lr":    args.encoder_lr,
            "head_lr":       args.head_lr,
            "pos_weight_cap": args.pos_weight_cap,
            "num_classes":   num_classes,
            "min_fact_chars_exclusive": args.min_fact_chars,
            "min_label_freq_exclusive": args.min_label_freq,
        },
    }
    result_path = out_dir / "test_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"結果已儲存：{result_path}")


if __name__ == "__main__":
    main()