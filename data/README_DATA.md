# SJTU SEED 数据集放置说明

因上海交通大学 BCMI 实验室 SEED 数据集受学术许可协议约束，本仓库不包含原始 `.mat` 数据文件。请向官方申请获取数据后，将 `ExtractedFeatures/` 放置于以下路径：

```text
data/
├── README_DATA.md
└── SEED/
    └── ExtractedFeatures/
        ├── label.mat
        └── {subject_id}_{date}.mat    # 15 名被试 × 3 次采集共 45 个 .mat 文件
```
