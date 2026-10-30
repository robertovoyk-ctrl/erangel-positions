"""Train survival prediction model for phases 2-8."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

PROCESSED = Path("data/processed")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "dist_to_blue_center",
    "dist_to_white_center",
    "in_blue",
    "in_white",
    "zone_edge_dist",
    "zone_edge_ratio",
    "elevation",
    "slope",
    "team_spread",
    "team_alive",
    "blue_r",
    "white_r",
    "phase",
]


def main():
    features_file = PROCESSED / "features.json"
    if not features_file.exists():
        print("No features.json found. Run process_data.py first.")
        return

    with open(features_file) as f:
        rows = json.load(f)

    df = pd.DataFrame(rows)
    print(f"Total rows: {len(df)}")
    print(f"Phases: {sorted(df['phase'].unique())}")
    print(f"Matches: {df['match_id'].nunique()}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["survived"])
    print(f"After cleaning: {len(df)} rows")

    X = df[FEATURE_COLS].values
    y = df["survived"].values

    print(f"\nClass balance: {y.mean():.3f} survival rate")
    print(f"Features: {len(FEATURE_COLS)}")

    # Split by MATCH to prevent leakage
    groups = df["match_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Fit scaler on TRAINING data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    train_matches = set(df.iloc[train_idx]["match_id"])
    test_matches = set(df.iloc[test_idx]["match_id"])
    print(f"\nMatch-based split: {len(train_matches)} train / {len(test_matches)} test matches")
    print(f"Rows: {len(train_idx)} train / {len(test_idx)} test")
    print(f"Overlap: {len(train_matches & test_matches)} matches")

    # Gradient Boosting
    print("\n=== Gradient Boosting ===")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_proba_test = gb.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_proba_test)
    print(f"Test accuracy: {gb_acc:.4f}")
    print(f"Test ROC AUC: {gb_auc:.4f}")

    gkf = GroupKFold(n_splits=5)
    gb_pipe = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
    )
    cv_scores = cross_val_score(
        gb_pipe, X_train_raw, y_train, cv=gkf, groups=groups_train, scoring="accuracy"
    )
    print(f"5-fold GroupKFold CV (train): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nFeature importance:")
    importances = list(zip(FEATURE_COLS, gb.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {name:25s} {imp:.4f} {bar}")

    # Logistic Regression
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_proba_test = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba_test)
    print(f"Test accuracy: {lr_acc:.4f}")
    print(f"Test ROC AUC: {lr_auc:.4f}")

    best_model = gb if gb_acc >= lr_acc else lr
    best_name = "GradientBoosting" if gb_acc >= lr_acc else "LogisticRegression"
    best_acc = max(gb_acc, lr_acc)
    best_auc = gb_auc if gb_acc >= lr_acc else lr_auc
    print(f"\nBest model: {best_name} ({best_acc:.4f})")

    # Baseline: always predict majority class
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    overall_lift = best_acc - baseline_acc
    print(f"\nBaseline (always-survive): {baseline_acc:.4f}")
    print(f"Lift over baseline: +{overall_lift:.4f}")
    print(f"Overall ROC AUC: {best_auc:.4f}")

    # Per-phase metrics on TEST SET ONLY
    print("\nPer-phase metrics (test set only):")
    phase_metrics = {}
    df_test = df.iloc[test_idx]
    for phase in sorted(df_test["phase"].unique()):
        phase_mask = df_test["phase"].values == phase
        n_phase = int(phase_mask.sum())
        if n_phase == 0:
            continue
        y_p = y_test[phase_mask]
        X_p = X_test[phase_mask]

        phase_baseline = max(y_p.mean(), 1 - y_p.mean())
        pred_p = best_model.predict(X_p)
        acc_p = accuracy_score(y_p, pred_p)

        try:
            proba_p = best_model.predict_proba(X_p)[:, 1]
            auc_p = roc_auc_score(y_p, proba_p)
        except ValueError:
            auc_p = 0.5

        lift_p = acc_p - phase_baseline
        phase_metrics[str(phase)] = {
            "baseline": round(float(phase_baseline), 4),
            "accuracy": round(float(acc_p), 4),
            "auc": round(float(auc_p), 4),
            "lift": round(float(lift_p), 4),
            "n_samples": n_phase,
            "survival_rate": round(float(y_p.mean()), 4),
        }

        flag = " !" if lift_p < 0.05 else ""
        print(f"  Phase {phase}: acc={acc_p:.3f} base={phase_baseline:.3f} "
              f"lift=+{lift_p:.3f} AUC={auc_p:.3f} (n={n_phase}){flag}")

    # Save models
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save train match IDs so downstream steps use training data only
    with open(MODEL_DIR / "train_matches.json", "w") as f:
        json.dump(sorted(train_matches), f)
    print(f"Saved {len(train_matches)} train match IDs")

    # LR coefficients for browser-side scoring (scaler fit on training only)
    lr_params = {
        "coefs": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "features": FEATURE_COLS,
    }

    # Per-phase median feature values from TRAINING data only
    df_train = df.iloc[train_idx]
    phase_medians = {}
    for phase in sorted(df_train["phase"].unique()):
        pdf = df_train[df_train["phase"] == phase]
        phase_medians[str(phase)] = {
            col: round(float(pdf[col].median()), 2) for col in FEATURE_COLS
        }

    model_info = {
        "model_type": best_name,
        "accuracy": round(best_acc, 4),
        "auc": round(float(best_auc), 4),
        "baseline_accuracy": round(float(baseline_acc), 4),
        "lift": round(float(overall_lift), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "features": FEATURE_COLS,
        "feature_importance": {n: round(float(v), 4) for n, v in importances} if best_name == "GradientBoosting" else {},
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_matches": len(train_matches),
        "n_test_matches": len(test_matches),
        "n_total": len(df),
        "n_matches": int(df["match_id"].nunique()),
        "class_balance": round(float(y.mean()), 4),
        "phase_metrics": phase_metrics,
        "lr_params": lr_params,
        "phase_medians": phase_medians,
    }
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to {MODEL_DIR}/")

    # Generate prediction grid for viewer (training data distributions)
    print("\nGenerating prediction grid for viewer...")
    generate_prediction_grid(best_model, scaler, df_train)


def generate_prediction_grid(model, scaler, df):
    grid_data = {}

    for phase in sorted(df["phase"].unique()):
        phase_df = df[df["phase"] == phase]
        if len(phase_df) < 10:
            continue

        median_features = phase_df[FEATURE_COLS].median()
        median_white_r = median_features["white_r"]
        median_white_x = phase_df["centroid_x"].median()
        median_white_y = phase_df["centroid_y"].median()

        grid_res = 50
        xs = np.linspace(0, 816000, grid_res)
        ys = np.linspace(0, 816000, grid_res)

        grid_probs = np.zeros((grid_res, grid_res))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                feat = median_features.copy()
                dist = np.sqrt((x - median_white_x)**2 + (y - median_white_y)**2)
                feat["dist_to_white_center"] = dist
                feat["zone_edge_dist"] = median_white_r - dist
                feat["zone_edge_ratio"] = dist / max(median_white_r, 1)
                feat["in_white"] = 1 if dist < median_white_r else 0
                feat["phase"] = phase

                X_grid = scaler.transform([feat[FEATURE_COLS].values])
                prob = model.predict_proba(X_grid)[0][1]
                grid_probs[j, i] = prob

        grid_data[str(phase)] = {
            "probs": grid_probs.tolist(),
            "white_x": float(median_white_x),
            "white_y": float(median_white_y),
            "white_r": float(median_white_r),
        }

    with open(PROCESSED / "prediction_grid.json", "w") as f:
        json.dump(grid_data, f)
    print(f"Prediction grid saved for phases: {list(grid_data.keys())}")


if __name__ == "__main__":
    main()
