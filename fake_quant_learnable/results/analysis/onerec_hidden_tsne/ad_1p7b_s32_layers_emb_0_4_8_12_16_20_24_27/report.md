# OneRec Layer-wise Hidden t-SNE

## 设置

- 模型：`/home/yhhuang/.cache/huggingface/hub/models--OpenOneRec--OneRec-1.7B/snapshots/OneRec-1.7B`
- 数据：`data/onerec_data/benchmark-data-calib1024`, split=`calib`, samples=`32`
- 层：`embedding, layer_00, layer_04, layer_08, layer_12, layer_16, layer_20, layer_24, layer_27`
- 采样 token：`{"text": 2000, "history_sid": 2000, "interest_sid": 2000, "sid_boundary": 1200}`
- t-SNE：PCA(50) -> t-SNE(perplexity=45.0, seed=42)

## 指标

| layer     |   points |   silhouette_tsne |   silhouette_pca20 |   pca_var_sum |
|:----------|---------:|------------------:|-------------------:|--------------:|
| embedding |     7200 |         0.0367172 |        0.0238294   |      0.560881 |
| layer_00  |     7200 |         0.159162  |        0.0513256   |      0.482142 |
| layer_04  |     7200 |         0.14386   |       -0.302111    |      0.997953 |
| layer_08  |     7200 |         0.105412  |       -0.244241    |      0.995116 |
| layer_12  |     7200 |         0.117257  |       -0.140164    |      0.985649 |
| layer_16  |     7200 |         0.161538  |        0.000468326 |      0.946747 |
| layer_20  |     7200 |         0.152941  |        0.0803823   |      0.817402 |
| layer_24  |     7200 |         0.167366  |        0.0680784   |      0.708717 |
| layer_27  |     7200 |         0.126854  |        0.106386    |      0.635246 |

## 文件

- `hidden_tsne_grid.png`
- `layer_metrics.csv`
- `sampled_token_points.csv`
- `points_<layer>.csv`
- `config.json`
