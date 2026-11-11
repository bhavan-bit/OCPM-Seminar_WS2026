"""Tabular baseline and tree models across Tiers 1, 2, and 3.

Models:
- OLS Linear Regression
- Regularized Linear: RidgeCV / LassoCV tuned on 20% validation split
- XGBoost: objective="reg:absoluteerror", early_stopping_rounds=50
- CatBoost: loss_function="MAE", early_stopping_rounds=50

Evaluated on fixed train/val/test splits from data/splits.json.
Logs both Test MAE and wall-clock training time.
"""

import argparse


def train_and_evaluate(tier: int, model_name: str, master_parquet: str, splits_json: str):
    """Trains a tabular model on the specified feature tier and logs results."""
    print(f"Training {model_name} on Tier {tier} features...")
    # TODO: Load master parquet, filter columns for tier, load splits, fit & evaluate
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tabular Models")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=1, help="Feature tier")
    parser.add_argument(
        "--model", choices=["ols", "ridge", "lasso", "xgboost", "catboost"], default="xgboost"
    )
    parser.add_argument("--data", default="data/processed/bpic17_tier3_master.parquet")
    parser.add_argument("--splits", default="data/splits.json")
    args = parser.parse_args()

    train_and_evaluate(args.tier, args.model, args.data, args.splits)
