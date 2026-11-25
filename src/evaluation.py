"""Within-Subject and Cross-Subject (LOSO) evaluation & ablation engine for SEED DGCNN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.data_pipeline import load_seed_dataset
from src.domain_adaptation import predict_pytorch_classifier, train_pytorch_classifier
from src.graph_utils import build_initial_adjacency, extract_region_connectivity_stats
from src.models import DGCNNNet, MLPNet, SVMClassifier, TSceptionNet
from src.utils import compute_emotion_metrics, save_json

logger = logging.getLogger("seed_dgcnn")


def _write_benchmark_report(
    within_df: pd.DataFrame,
    loso_df: pd.DataFrame,
    region_df: pd.DataFrame,
    n_subjects: int,
    out_path: Path,
) -> None:
    lines = [
        "# SEED（62导三分类）动态图神经网络（DGCNN）与跨被试域适应汇总",
        "",
        f"- 评估被试数：`{n_subjects}` 名（被试内 9/6 试次划分 + 跨被试严格 LOSO）",
        "",
    ]
    if not within_df.empty:
        lines.extend(["## 1. 被试内（Within-Subject）三分类评估汇总", "", within_df.to_markdown(index=False), ""])
        best_w = within_df.sort_values("accuracy_pct_mean", ascending=False).iloc[0]
        lines.append(
            f"> **被试内最优结果（{best_w['model']}）**：三分类准确率 **{best_w['accuracy_pct_mean']:.1f}%** "
            f"（Macro-F1 **{best_w['macro_f1_mean']:.3f}**）。"
        )
        lines.append("")

    if not loso_df.empty:
        lines.extend(["## 2. 跨被试（LOSO）域适应消融汇总", "", loso_df.to_markdown(index=False), ""])
        best_l = loso_df.sort_values("accuracy_pct_mean", ascending=False).iloc[0]
        base_l = loso_df[loso_df["model"] == "DGCNN_NoDA"]
        if not base_l.empty:
            gain = best_l["accuracy_pct_mean"] - base_l.iloc[0]["accuracy_pct_mean"]
            lines.append(
                f"> **跨被试最优结果（{best_l['model']}）**：较未对齐 DGCNN 基线（{base_l.iloc[0]['accuracy_pct_mean']:.1f}%）"
                f"提升 **{gain:+.1f}%** 至 **{best_l['accuracy_pct_mean']:.1f}%**（Macro-F1 **{best_l['macro_f1_mean']:.3f}**）。"
            )
        lines.append("")

    if not region_df.empty:
        lines.extend(["## 3. DGCNN 学习到的脑区功能连接贡献度排名", "", region_df.to_markdown(index=False), ""])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_seed_benchmark(
    target_subjects: Sequence[int],
    cfg: Dict[str, Any],
    run_within: bool = True,
    run_loso: bool = True,
) -> Dict[str, Any]:
    results_dir = Path(cfg["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_subjects = [int(s) for s in cfg["dataset"]["subjects"]]
    seed_data = load_seed_dataset(all_subjects, cfg)

    gcfg = cfg["graph"]
    tcfg = cfg["training"]
    n_classes = int(cfg["dataset"]["n_classes"])
    in_features = seed_data[all_subjects[0]]["X_all"].shape[-1]
    init_adj = build_initial_adjacency(sigma_scale=float(gcfg["sigma_scale"]))

    within_rows: List[Dict[str, Any]] = []
    loso_rows: List[Dict[str, Any]] = []
    learned_adjs: List[np.ndarray] = []

    # 1. Within-subject evaluation (official 9 train / 6 test trials)
    if run_within:
        logger.info("=== Starting Within-Subject Evaluation (%d subjects) ===", len(target_subjects))
        for subj in target_subjects:
            subj = int(subj)
            X_tr = seed_data[subj]["X_within_train"]
            y_tr = seed_data[subj]["y_within_train"]
            X_te = seed_data[subj]["X_within_test"]
            y_te = seed_data[subj]["y_within_test"]

            # SVM
            svm = SVMClassifier().fit(X_tr, y_tr)
            within_rows.append({"subject": subj, "model": "SVM", **compute_emotion_metrics(y_te, svm.predict(X_te))})

            # PyTorch models: MLP, TSception, DGCNN
            py_models = [
                ("MLP", MLPNet(62, in_features, gcfg["latent_dim"], n_classes, gcfg["dropout"])),
                ("TSception", TSceptionNet(62, in_features, gcfg["latent_dim"], n_classes, gcfg["dropout"])),
                (
                    "DGCNN",
                    DGCNNNet(
                        init_adj=init_adj,
                        in_features=in_features,
                        hidden_dim=gcfg["hidden_dim"],
                        latent_dim=gcfg["latent_dim"],
                        cheb_k=gcfg["cheb_k"],
                        n_classes=n_classes,
                        dropout=gcfg["dropout"],
                    ),
                ),
            ]
            for name, net in py_models:
                net = train_pytorch_classifier(
                    net,
                    X_tr,
                    y_tr,
                    da_mode="none",
                    epochs=int(tcfg["within_epochs"]),
                    batch_size=int(tcfg["batch_size"]),
                    lr=float(tcfg["lr"]),
                    weight_decay=float(tcfg["weight_decay"]),
                    latent_dim=int(gcfg["latent_dim"]),
                )
                preds = predict_pytorch_classifier(net, X_te)
                m = compute_emotion_metrics(y_te, preds)
                within_rows.append({"subject": subj, "model": name, **m})
                if name == "DGCNN":
                    learned_adjs.append(net.get_learned_adjacency())
            pd.DataFrame(within_rows).to_csv(results_dir / "within_per_subject.csv", index=False)
            logger.info("[Within] Subject %02d done.", subj)

    # 2. Cross-subject LOSO evaluation with MMD / DANN ablations
    if run_loso:
        logger.info("=== Starting Cross-Subject LOSO Evaluation (%d target subjects) ===", len(target_subjects))
        for test_subj in target_subjects:
            test_subj = int(test_subj)
            train_subjs = [s for s in all_subjects if s != test_subj]
            X_src = np.concatenate([seed_data[s]["X_all"] for s in train_subjs], axis=0)
            y_src = np.concatenate([seed_data[s]["y_all"] for s in train_subjs], axis=0)
            X_tgt = seed_data[test_subj]["X_all"]
            y_tgt = seed_data[test_subj]["y_all"]

            # SVM baseline
            svm = SVMClassifier().fit(X_src, y_src)
            loso_rows.append({"target_subject": test_subj, "model": "SVM", **compute_emotion_metrics(y_tgt, svm.predict(X_tgt))})

            loso_specs = [
                ("MLP", lambda: MLPNet(62, in_features, gcfg["latent_dim"], n_classes, gcfg["dropout"]), "none"),
                ("TSception", lambda: TSceptionNet(62, in_features, gcfg["latent_dim"], n_classes, gcfg["dropout"]), "none"),
                ("DGCNN_NoDA", lambda: DGCNNNet(init_adj, in_features, gcfg["hidden_dim"], gcfg["latent_dim"], gcfg["cheb_k"], n_classes, gcfg["dropout"]), "none"),
                ("DGCNN_MMD", lambda: DGCNNNet(init_adj, in_features, gcfg["hidden_dim"], gcfg["latent_dim"], gcfg["cheb_k"], n_classes, gcfg["dropout"]), "mmd"),
                ("DGCNN_DANN", lambda: DGCNNNet(init_adj, in_features, gcfg["hidden_dim"], gcfg["latent_dim"], gcfg["cheb_k"], n_classes, gcfg["dropout"]), "dann"),
                ("DGCNN_MMD_DANN", lambda: DGCNNNet(init_adj, in_features, gcfg["hidden_dim"], gcfg["latent_dim"], gcfg["cheb_k"], n_classes, gcfg["dropout"]), "mmd_dann"),
            ]

            for name, builder, da_mode in loso_specs:
                net = train_pytorch_classifier(
                    builder(),
                    X_src,
                    y_src,
                    X_target_unlabeled=X_tgt if da_mode != "none" else None,
                    da_mode=da_mode,
                    epochs=int(tcfg["loso_epochs"]),
                    batch_size=int(tcfg["batch_size"]),
                    lr=float(tcfg["lr"]),
                    weight_decay=float(tcfg["weight_decay"]),
                    mmd_weight=float(tcfg["mmd_weight"]),
                    dann_weight=float(tcfg["dann_weight"]),
                    latent_dim=int(gcfg["latent_dim"]),
                )
                preds = predict_pytorch_classifier(net, X_tgt)
                m = compute_emotion_metrics(y_tgt, preds)
                loso_rows.append({"target_subject": test_subj, "model": name, **m})
                logger.info("[LOSO Subj %02d] %-16s | Acc: %5.2f%% | F1: %.4f", test_subj, name, m["accuracy_pct"], m["macro_f1"])
                if name == "DGCNN_MMD_DANN":
                    learned_adjs.append(net.get_learned_adjacency())

            pd.DataFrame(loso_rows).to_csv(results_dir / "loso_per_subject.csv", index=False)

    within_summary = pd.DataFrame()
    if within_rows:
        df_w = pd.DataFrame(within_rows)
        within_summary = (
            df_w.groupby("model", as_index=False)
            .agg(
                accuracy_pct_mean=("accuracy_pct", "mean"),
                accuracy_pct_std=("accuracy_pct", "std"),
                macro_f1_mean=("macro_f1", "mean"),
            )
            .round(3)
        )
        within_summary.to_csv(results_dir / "within_summary.csv", index=False)

    loso_summary = pd.DataFrame()
    if not loso_rows and (results_dir / "loso_per_subject.csv").exists():
        loso_rows = pd.read_csv(results_dir / "loso_per_subject.csv").to_dict(orient="records")
    if loso_rows:
        df_l = pd.DataFrame(loso_rows)
        loso_summary = (
            df_l.groupby("model", as_index=False)
            .agg(
                accuracy_pct_mean=("accuracy_pct", "mean"),
                accuracy_pct_std=("accuracy_pct", "std"),
                macro_f1_mean=("macro_f1", "mean"),
            )
            .round(3)
        )
        loso_summary.to_csv(results_dir / "loso_summary.csv", index=False)

    region_df = pd.DataFrame()
    if learned_adjs:
        mean_adj = np.mean(learned_adjs, axis=0)
        np.save(results_dir / "dgcnn_learned_adjacency_mean.npy", mean_adj)
        region_df = pd.DataFrame(extract_region_connectivity_stats(mean_adj, A_init=init_adj))
        region_df.to_csv(results_dir / "dgcnn_region_connectivity.csv", index=False)

    payload = {
        "target_subjects": [int(s) for s in target_subjects],
        "within_summary": within_summary.to_dict(orient="records"),
        "loso_summary": loso_summary.to_dict(orient="records"),
        "region_connectivity": region_df.to_dict(orient="records"),
    }
    save_json(payload, results_dir / "experiment_summary.json")
    _write_benchmark_report(within_summary, loso_summary, region_df, len(target_subjects), results_dir / "benchmark_report.md")
    logger.info("All SEED results saved to %s", results_dir)
    return payload
