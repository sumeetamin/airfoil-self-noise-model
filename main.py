"""Select and evaluate models for the UCI Airfoil Self-Noise dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_data() -> tuple[pd.DataFrame, pd.Series, str]:
    dataset = fetch_ucirepo(id=291)
    X = dataset.data.features.apply(pd.to_numeric, errors="coerce")
    y_frame = dataset.data.targets.apply(pd.to_numeric, errors="coerce")
    if y_frame.shape[1] != 1:
        raise ValueError(f"Expected one continuous target, got {list(y_frame.columns)}")
    y = y_frame.iloc[:, 0]
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), str(y.name)


def make_model(estimator) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)])


def train(output_dir: Path) -> dict[str, object]:
    X, y, target = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    candidates = {
        "ridge": make_model(Ridge(alpha=10.0)),
        "random_forest": make_model(RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1)),
        "hist_gradient_boosting": make_model(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=42)),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    selection = {}
    for name, candidate in candidates.items():
        scores = -cross_val_score(candidate, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        selection[name] = {"cv_rmse_mean": float(scores.mean()), "cv_rmse_std": float(scores.std())}
    winner_name = min(selection, key=lambda name: selection[name]["cv_rmse_mean"])
    winner = candidates[winner_name].fit(X_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    pred, base_pred = winner.predict(X_test), baseline.predict(X_test)
    metrics = {
        "dataset": "UCI Airfoil Self-Noise (id 291)",
        "rows": int(len(X)),
        "target": target,
        "split": "75/25 random holdout; seed=42; CV selection restricted to training partition",
        "candidate_cv_rmse": selection,
        "selected_model": winner_name,
        "holdout": {
            "model": {"mae": float(mean_absolute_error(y_test, pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, pred))), "r2": float(r2_score(y_test, pred))},
            "training_mean_baseline": {"mae": float(mean_absolute_error(y_test, base_pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, base_pred))), "r2": float(r2_score(y_test, base_pred))},
        },
        "limitations": "Random row split; unseen-airfoil and campaign-level generalization are not established.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred, "baseline_predicted": base_pred}).to_csv(output_dir / "holdout_predictions.csv", index=False)
    importance = permutation_importance(winner, X_test, y_test, n_repeats=20, random_state=42, scoring="neg_root_mean_squared_error", n_jobs=-1)
    pd.DataFrame({"feature": X.columns, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False).to_csv(output_dir / "permutation_importance.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(winner, output_dir / "model.joblib")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    print(json.dumps(train(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

