# OneRec-1.7B Embedding Linear Probe

## 设置

- 输入：OneRec input embedding，维度 `2048`
- Token 数：`49152`
- 计数：`{"text": 24576, "s_a": 8192, "s_b": 8192, "s_c": 8192}`
- Probe：StandardScaler + SGDClassifier(log_loss, class_weight=balanced)
- 划分：Stratified train/test, test_size=0.3, seed=42

## Binary: SID vs Text

- Accuracy: `0.9796`
- Balanced accuracy: `0.9796`
- ROC-AUC: `0.9905`

Confusion matrix label order: `[text=0, SID=1]`

```json
[
  [
    7186,
    187
  ],
  [
    114,
    7259
  ]
]
```

## Multiclass: text / s_a / s_b / s_c

Label order: `['s_a', 's_b', 's_c', 'text']`

- Accuracy: `0.9275`
- Balanced accuracy: `0.9020`

```json
[
  [
    2284,
    18,
    7,
    148
  ],
  [
    0,
    1746,
    643,
    69
  ],
  [
    0,
    1,
    2433,
    24
  ],
  [
    44,
    70,
    45,
    7214
  ]
]
```

## 辅助文件

- `probe_results.json`
- `center_radius.csv`
- `center_distances.csv`
- `probe_tokens.csv`
