"""Train survival prediction model for phases 4-8."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

PROCESSED = Path("data/processed")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "dist_to_white_center",
    "dist_to_blue_center",
    "in_white",
    "in_blue",
    "zone_edge_dist",
    "zone_edge_ratio",
    "elevation",
    "slope",
    "team_spread",
    "team_alive",
    "white_r",
    "blue_r",
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

    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["survived"])
    print(f"After cleaning: {len(df)} rows")

    X = df[FEATURE_COLS].values
    y = df["survived"].values

    print(f"\nClass balance: {y.mean():.3f} survival rate")
    print(f"Features: {len(FEATURE_COLS)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient Boosting
    print("\n=== Gradient Boosting ===")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"Test accuracy: {gb_acc:.4f}")

    cv_scores = cross_val_score(gb, X_scaled, y, cv=5, scoring="accuracy")
    print(f"5-fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nFeature importance:")
    importances = list(zip(FEATURE_COLS, gb.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"  {name:25s} {imp:.4f} {bar}")

    # Logistic Regression for comparison
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Test accuracy: {lr_acc:.4f}")

    # Use the better model
    best_model = gb if gb_acc >= lr_acc else lr
    best_name = "GradientBoosting" if gb_acc >= lr_acc else "LogisticRegression"
    best_acc = max(gb_acc, lr_acc)
    print(f"\nBest model: {best_name} ({best_acc:.4f})")

    # Per-phase accuracy
    print("\nPer-phase accuracy (best model):")
    for phase in sorted(df["phase"].unique()):
        mask = df["phase"] == phase
        X_p = scaler.transform(df.loc[mask, FEATURE_COLS].values)
        y_p = df.loc[mask, "survived"].values
        pred_p = best_model.predict(X_p)
        acc_p = accuracy_score(y_p, pred_p)
        print(f"  Phase {phase}: {acc_p:.4f} ({mask.sum()} samples)")

    # Save model
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    model_info = {
        "model_type": best_name,
        "accuracy": best_acc,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "features": FEATURE_COLS,
        "feature_importance": {n: float(v) for n, v in importances} if best_name == "GradientBoosting" else {},
        "n_samples": len(df),
        "n_matches": int(df["match_id"].nunique()),
        "class_balance": float(y.mean()),
    }
    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nModel saved to {MODEL_DIR}/")

    # Generate prediction grid for viewer
    print("\nGenerating prediction grid for viewer...")
    generate_prediction_grid(best_model, scaler, df)


def generate_prediction_grid(model, scaler, df):
    """Generate a grid of survival predictions for the viewer overlay."""
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
