# Airfoil Self-Noise Model

Model-selection workflow for estimating scaled sound pressure from airfoil geometry and wind-tunnel conditions. It compares a mean baseline, a regularized linear model, and two tree models using cross-validation on the training partition, then evaluates the selected model once on a held-out test set.

## Quickstart

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python main.py --output-dir artifacts
```

The run downloads the public dataset through `ucimlrepo` and writes `metrics.json`, `holdout_predictions.csv`, `permutation_importance.csv`, and `model.joblib` to the output directory.

## Dataset and limits

The [UCI Airfoil Self-Noise dataset](https://archive.ics.uci.edu/dataset/291/airfoil+self-noise) contains 1,503 wind-tunnel observations from NASA aerodynamic/acoustic tests (Brooks, Pope & Marcolini, 1989), licensed CC BY 4.0, DOI [10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C). Rows are not grouped by airfoil or test campaign in the supplied interface, so the random holdout estimates interpolation on this benchmark, not generalization to unseen airfoils or deployment conditions. This is an engineering research baseline, not a certification or safety tool.

## Evaluation design

- Split once into a 75% training partition and a 25% final holdout (seed 42).
- Select among candidates using five-fold CV RMSE within the training partition only.
- Compare final holdout MAE, RMSE, and R² against a training-mean baseline.
- Compute permutation importance on the holdout for descriptive analysis; importance is not causal evidence.

