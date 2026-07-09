# OneRec-1.7B Full-Text Input Embedding PCA

- Total plotted points: `176227`
- Counts: `{"text": 151649, "s_a": 8192, "s_b": 8192, "s_c": 8192, "sid_boundary": 2}`
- PCA20 silhouette: `0.0436`
- PCA20 explained variance sum: `0.0444`

Files:

- `embedding_pca_full_text_big.png`
- `embedding_pca_sid_only_big.png`
- `embedding_points_full_text_pca.csv`
- `group_centers_pca.csv`
- `config.json`

Note: this revision uses full text tokens. Full t-SNE for 176k points is not generated with sklearn because it is not suitable for quick iteration.
