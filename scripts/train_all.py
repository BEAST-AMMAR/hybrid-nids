"""
scripts/train_all.py
====================
One-shot training script: loads preprocessed data, trains all models,
saves them to models/ and prints a full evaluation report.

Usage:
    python scripts/train_all.py [--data-dir notebooks] [--model-dir models]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.utils import load_config, get_project_root
from src.preprocessing import load_preprocessed, load_raw, preprocess_df
from src.models.autoencoder import AutoencoderModel
from src.models.isolation_forest import IFModel
from src.models.hybrid_detector import HybridDetector
from src.models.agreement_layer import AgreementLayer


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    print(f"\n--- {label} ---")
    print(classification_report(y_true, y_pred, target_names=["normal", "attack"]))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:\n  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}")
    print(f"False Positive Rate: {fp / (fp + tn):.4f}")
    print(f"False Negative Rate: {fn / (fn + tp):.4f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(data_dir: str, model_dir: str) -> None:
    root = get_project_root()
    cfg  = load_config()

    model_dir_path = root / model_dir
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    print_section("Loading Preprocessed Data")
    data = load_preprocessed(root / data_dir)
    X_train        = data["X_train"]
    X_test         = data["X_test"]
    X_train_normal = data["X_train_normal"]
    y_train        = data["y_train"]
    y_test         = data["y_test"]

    print(f"X_train shape:         {X_train.shape}")
    print(f"X_test  shape:         {X_test.shape}")
    print(f"X_train_normal shape:  {X_train_normal.shape}")
    print(f"y_train positive rate: {y_train.mean():.3f}")
    print(f"y_test  positive rate: {y_test.mean():.3f}")

    # ---------------------------------------------------------------
    # 2. Autoencoder
    # ---------------------------------------------------------------
    print_section("Training Autoencoder")
    ae = AutoencoderModel(
        input_dim=X_train.shape[1],
        config=cfg.get("autoencoder", {}),
    )
    ae.build()
    ae.model.summary()
    ae.train(X_train_normal, verbose=1)

    # Fit AE score scaler on training set
    ae.score(X_train, fit_scaler=True)

    # ---------------------------------------------------------------
    # 3. Isolation Forest
    # ---------------------------------------------------------------
    print_section("Training Isolation Forest")
    ifm = IFModel(config=cfg.get("isolation_forest", {}))
    ifm.train(X_train)

    # Fit IF score scaler on training set
    ifm.score(X_train, fit_scaler=True)

    # ---------------------------------------------------------------
    # 4. Hybrid Detector — calibrate threshold on TEST set
    # ---------------------------------------------------------------
    print_section("Calibrating Hybrid Detector")
    ae_cfg = cfg.get("hybrid", {})
    hybrid = HybridDetector(
        ae_model  = ae,
        if_model  = ifm,
        ae_weight = ae_cfg.get("ae_weight", 0.6),
        if_weight = ae_cfg.get("if_weight", 0.4),
    )
    hybrid.calibrate_threshold(X_test, y_test)

    # Evaluate standalone hybrid (no agreement layer)
    y_pred_hybrid = hybrid.predict(X_test)
    evaluate(y_test, y_pred_hybrid, "Hybrid (AE + IF, no agreement layer)")

    # ---------------------------------------------------------------
    # 5. Agreement Layer — train XGBoost arbitrator
    # ---------------------------------------------------------------
    print_section("Training Agreement Layer (XGBoost Arbitrator)")
    agr_cfg = cfg.get("agreement", {})
    xgb_cfg = cfg.get("xgboost_arbitrator", {})
    layer = AgreementLayer(
        uncertainty_factor = agr_cfg.get("uncertainty_factor", 0.5),
        config             = xgb_cfg,
    )
    t = hybrid.threshold
    low, high = layer._band(t)
    print(f"Youden's threshold: {t:.4f}")
    print(f"Tier 1 band  → normal if score < {low:.4f}, attack if score > {high:.4f}")
    print(f"Tier 2 band  → uncertain if {low:.4f} ≤ score ≤ {high:.4f}")
    layer.train_arbitrator(X_train, y_train)

    # ---------------------------------------------------------------
    # 6. Full pipeline evaluation on test set
    # ---------------------------------------------------------------
    print_section("Full Pipeline Evaluation on Test Set")
    hybrid_scores_test, ae_scores_test, if_scores_test = hybrid.hybrid_score(X_test)
    results_df = layer.predict_df(
        X_test,
        hybrid_scores_test,
        ae_scores_test,
        if_scores_test,
        hybrid.threshold,
    )

    y_pred_full = results_df["final_pred"].values
    evaluate(y_test, y_pred_full, "Full Pipeline (Hybrid + Agreement Layer)")

    # Tier breakdown
    tier_counts = results_df["tier_used"].value_counts().sort_index()
    print("\nTier breakdown:")
    for tier, count in tier_counts.items():
        pct = count / len(results_df) * 100
        print(f"  Tier {tier}: {count:,} samples ({pct:.1f}%)")

    # Confidence breakdown
    conf_counts = results_df["confidence"].value_counts()
    print("\nConfidence breakdown:")
    for level, count in conf_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {level}: {count:,} samples ({pct:.1f}%)")

    # ---------------------------------------------------------------
    # 7. Save scaler + feature columns for raw inference
    # ---------------------------------------------------------------
    print_section("Saving Scaler and Feature Columns")
    raw_train = root / "data" / "KDDTest+.txt"       # NOTE: KDDTest+ is used as train in notebooks
    raw_test  = root / "data" / "KDDTrain+_20Percent.txt"
    if raw_train.exists() and raw_test.exists():
        print("Re-running preprocessing on raw data to fit scaler...")
        train_df = load_raw(raw_train)
        test_df  = load_raw(raw_test)
        _, _, _, _, scaler, feature_columns = preprocess_df(train_df, test_df)
        joblib.dump(scaler, model_dir_path / "scaler.pkl")
        joblib.dump(feature_columns, model_dir_path / "feature_columns.pkl")
        print(f"Scaler saved to models/scaler.pkl ({len(feature_columns)} features)")
    else:
        print("Raw data not found — skipping scaler.pkl (detect_raw/detect_single will be unavailable)")

    # ---------------------------------------------------------------
    # 8. Save all models
    # ---------------------------------------------------------------
    print_section("Saving Models")
    hybrid.save(model_dir_path)
    layer.save(model_dir_path)
    print(f"All models saved to: {model_dir_path}")
    print("Files written:")
    for f in sorted(model_dir_path.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:>8.1f} KB")

    print_section("Training Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all Hybrid NIDS models")
    parser.add_argument("--data-dir",  default="notebooks",
                        help="Directory containing .npy preprocessed files")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save trained models")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir)
