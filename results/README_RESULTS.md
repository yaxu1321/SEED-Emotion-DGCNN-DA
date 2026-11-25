# 实验结果文件说明

本目录包含 SEED 数据集全部 15 名被试在被试内（9/6 试次划分）与跨被试（15 折 LOSO）双协议下的评测输出：
- `benchmark_report.md`：双协议模型对比、域适应消融与脑区连接塑性分析报告
- `within_summary.csv` / `within_per_subject.csv`：SVM、MLP、TSception、DGCNN 被试内三分类评估结果
- `loso_summary.csv` / `loso_per_subject.csv`：15 折 LOSO 下无域适应与 MK-MMD / DANN 域适应消融结果
- `dgcnn_region_connectivity.csv` & `dgcnn_learned_adjacency_mean.npy`：可学习邻接矩阵均值及五大脑区正向塑性增益统计
- `experiment_summary.json`：结构化汇总指标
