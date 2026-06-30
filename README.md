# CAIL2018 罪名多標籤預測｜ELECTRA-base Fine-tuning

基於 `hfl/chinese-legal-electra-base-discriminator` 對 CAIL2018 資料集進行罪名多標籤分類實驗。

## 分支說明

| 分支 | 說明 |
|---|---|
| `main` | 主實驗：全參數 fine-tuning，202 類罪名，8 epoch |
| `compare` | 對照實驗：照組員統一設定，160 類罪名，1 epoch，用於跨模型比較 |

---

## 主實驗結果（main 分支）

| 指標 | Valid（Epoch 8） | Test |
|---|---|---|
| Micro F1 | 0.8577 | 0.8318 |
| Macro F1 | 0.7663 | 0.7476 |
| Weighted F1 | 0.8642 | 0.8393 |
| Samples F1 | 0.8842 | 0.8599 |

> ⚠️ 報告中呈現的數字以 **Test set** 為準，Valid set 僅用於早停與模型選擇。

---

## 實驗結果（compare 分支）

與組員僅改動骨幹模型大小，其餘（pooling、dropout、pos_weight、freeze 策略）皆對齊：

|  | 本實驗（ELECTRA-base） |
|---|---|
| 類別數 | 160 |
| 訓練模式 | 全參數 fine-tuning |
| Epoch | 1 |
| Pooling | Attention Pooling |
| Dropout | 0.1 |
| pos_weight | min(neg/pos, 10) |
| Valid Macro F1 | 0.7361 |
| Test Macro F1  | 0.7239 |
| Test Micro F1 | 0.7590 |

> 僅 1 epoch 下未充分收斂，目的是在控制變因的前提下比較模型規模對學習速度的影響。

---

## 模型架構

```
輸入文字（fact，最長 512 tokens）
        ↓
ELECTRA-base Encoder（全參數更新）
hfl/chinese-legal-electra-base-discriminator
12 layers, hidden=768
        ↓
CLS token（768 維）
        ↓
Dropout(0.2)
        ↓
Linear(768 → 202)
        ↓
BCEWithLogitsLoss + log1p pos_weight
```

---

## 訓練設定（main 分支）

| 參數 | 值 |
|---|---|
| 模型 | hfl/chinese-legal-electra-base-discriminator |
| 訓練模式 | 全參數 fine-tuning（解凍） |
| 罪名類別數 | 202（原始 CAIL2018） |
| Loss | BCEWithLogitsLoss + log1p pos_weight |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 2（Gradient Accumulation 8 步，等效 batch 16） |
| Epochs | 8 |
| Threshold | 0.5 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU（4GB VRAM） |

---

## 資料集

CAIL2018（中國法研杯司法人工智能挑战赛）

| 分割 | 筆數 |
|---|---|
| Train | 154,592 |
| Valid | 17,131 |
| Test | 32,508 |

資料來源：[CAIL2018 官方](https://github.com/china-ai-law-challenge/CAIL2018)

> 資料檔案不包含在此 repo 中。

---

## 環境安裝

```bash
pip install -r requirements.txt
```

---

## 使用方式

**主實驗訓練（main 分支）：**
```bash
python script.py
```

**對照實驗訓練（compare 分支）：**
```bash
python script_compare.py
```

路徑設定在各腳本的 `class Args` 內，依實際資料位置修改。

---

## 已知限制

- 輸入文字超過 512 tokens 的案件（約 24%）會被截斷，後半段資訊遺失
- Recall 高於 Precision 為 pos_weight 的預期副作用，可透過調整 threshold 改善
- 凍結版 ELECTRA 表現較差，因為 ELECTRA 預訓練任務（RTD）為 token 層級，[CLS] 未被訓練來代表句子語意