# OneRec-1.7B Input Embedding Visualization

## 设置

- 模型：`/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B`
- 点数：`5502`
- 分组：`s_a`, `s_b`, `s_c`, `sid_boundary`, `text`
- 采样：每类 SID 最多 `1000`，text 最多 `2500`
- 降维：PCA(50) 后 t-SNE(perplexity=50.0, seed=42)

## 计数

```json
{
  "s_a": 1000,
  "s_b": 1000,
  "s_c": 1000,
  "sid_boundary": 2,
  "text": 2500
}
```

## 定量参考

- Silhouette on PCA20：`0.0516`
- Silhouette on t-SNE2：`-0.0117`
- PCA 前 50 维解释方差：`0.1302`

## 文件

- `embedding_tsne_by_token_type.png`
- `embedding_pca_by_token_type.png`
- `embedding_points.csv`
- `group_centers.csv`
- `config.json`
