"""SHAP-based Recursive Feature Elimination (SHAP-RFE) for OCPM.

This module provides a production-grade, modular feature selection pipeline
designed for Object-Centric Predictive Process Monitoring (OCPM). It addresses
high feature collinearity (e.g., between Synchronization Time P5 and Waiting Time P10)
and uses out-of-fold validation SHAP values to eliminate uninformative features.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class SHAP_RFE_Result:
    """Encapsulates results from the SHAP-RFE pipeline.

    Attributes:
        best_features (List[str]): Feature subset that achieved global minimum validation MAE.
        best_mae (float): Global minimum validation MAE.
        best_model (xgb.XGBRegressor): XGBoost model trained on the best feature subset.
        history (pd.DataFrame): Iteration-level audit log of feature counts, MAE, and dropped features.
        dropped_collinear_features (List[str]): Features removed during the correlation filter.
    """

    best_features: List[str]
    best_mae: float
    best_model: xgb.XGBRegressor
    history: pd.DataFrame
    dropped_collinear_features: List[str]


def remove_highly_correlated_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    threshold: float = 0.85,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Filters collinear features using Pearson correlation on training data.

    Computes an absolute Pearson correlation matrix on X_train. For any pair of
    features with |r| > threshold, drops the redundant feature (second occurrence
    in upper triangle) to reduce matrix instability and SHAP dilution.

    Args:
        X_train: Training feature DataFrame.
        X_val: Validation feature DataFrame.
        threshold: Absolute Pearson correlation cutoff (default: 0.85).

    Returns:
        Tuple containing:
            - Filtered X_train DataFrame.
            - Filtered X_val DataFrame.
            - List of dropped feature names.

    Raises:
        ValueError: If threshold is not within (0, 1].
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"Correlation threshold must be in (0, 1], got {threshold}")

    logger.info("Computing Pearson correlation matrix on training set (threshold=%.2f)...", threshold)
    corr_matrix = X_train.corr(method="pearson").abs()

    # Mask upper triangle to inspect unique pairs without self-correlation
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop: List[str] = [
        col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)
    ]

    for col in to_drop:
        # Find which feature(s) it correlated with for logging
        correlated_with = upper_triangle.index[upper_triangle[col] > threshold].tolist()
        logger.info(
            "Dropping collinear feature '%s' (correlated with %s, |r| > %.2f)",
            col,
            correlated_with,
            threshold,
        )

    X_train_filtered = X_train.drop(columns=to_drop)
    X_val_filtered = X_val.drop(columns=to_drop)

    logger.info(
        "Collinearity filter complete: %d initial features -> %d retained (%d dropped)",
        X_train.shape[1],
        X_train_filtered.shape[1],
        len(to_drop),
    )

    return X_train_filtered, X_val_filtered, to_drop


def calculate_val_shap_importance(
    model: xgb.XGBRegressor,
    X_val: pd.DataFrame,
) -> pd.Series:
    """Calculates mean absolute SHAP feature importance strictly on validation data.

    Using validation data prevents the explainer from capturing overfitted noise
    from training data, yielding more robust feature rankings for elimination.

    Args:
        model: Trained XGBoost regressor model.
        X_val: Validation feature DataFrame matching model features.

    Returns:
        pd.Series: Mean absolute SHAP values indexed by feature name, sorted ascending.
    """
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_val)

    # Standardize across different SHAP versions (Explanation vs ndarray)
    if hasattr(raw_shap, "values"):
        raw_shap = raw_shap.values

    # Mean absolute impact per feature across all validation rows
    mean_abs_impact = np.mean(np.abs(raw_shap), axis=0)
    shap_importance = pd.Series(mean_abs_impact, index=X_val.columns).sort_values(ascending=True)

    return shap_importance


def run_shap_rfe_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    correlation_threshold: float = 0.85,
    drop_fraction: float = 0.10,
    patience: int = 2,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> SHAP_RFE_Result:
    """Executes the complete SHAP-based Recursive Feature Elimination pipeline.

    Workflow:
        1. Filters collinear features (|r| > correlation_threshold) on X_train.
        2. Iteratively fits XGBRegressor on current X_train.
        3. Computes TreeSHAP feature importance strictly on X_val.
        4. Evaluates validation MAE and tracks global minimum.
        5. Drops bottom `drop_fraction` features based on mean absolute SHAP value.
        6. Stops when validation MAE increases for `patience` consecutive iterations.
        7. Retrains and returns the model on the best global feature subset.

    Args:
        X_train: Training feature DataFrame.
        y_train: Target remaining time Series for training.
        X_val: Validation feature DataFrame.
        y_val: Target remaining time Series for validation.
        correlation_threshold: Cutoff for removing collinear features (default: 0.85).
        drop_fraction: Fraction of bottom SHAP features to drop per step (default: 0.10).
        patience: Consecutive MAE degradation steps before early stopping (default: 2).
        xgb_params: Optional parameter dictionary passed to XGBRegressor.

    Returns:
        SHAP_RFE_Result: Named result containing best features, best MAE, best model,
                         iteration history, and collinear dropped features.

    Raises:
        ValueError: If input DataFrames are empty or feature sets mismatch.
    """
    if X_train.empty or X_val.empty:
        raise ValueError("X_train and X_val must be non-empty DataFrames.")
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("X_train and X_val must have identical columns.")

    # Base XGBoost parameters optimized for MAE in process monitoring
    default_xgb_params: Dict[str, Any] = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    if xgb_params:
        default_xgb_params.update(xgb_params)

    # Step 1: Pre-processing Collinearity Filter
    X_tr_curr, X_va_curr, dropped_collinear = remove_highly_correlated_features(
        X_train=X_train,
        X_val=X_val,
        threshold=correlation_threshold,
    )

    current_features = list(X_tr_curr.columns)
    best_features = list(current_features)
    best_mae = float("inf")
    best_model: Optional[xgb.XGBRegressor] = None

    history_records: List[Dict[str, Any]] = []
    prev_mae: Optional[float] = None
    consecutive_degradations: int = 0
    iteration: int = 1

    logger.info("Beginning SHAP-RFE loop with %d initial candidate features...", len(current_features))

    while len(current_features) > 1:
        logger.info(
            "--- Iteration %d: Training XGBoost with %d features ---",
            iteration,
            len(current_features),
        )

        # Train model completely from scratch on reduced subset
        model = xgb.XGBRegressor(**default_xgb_params)
        model.fit(
            X_tr_curr[current_features],
            y_train,
            eval_set=[(X_va_curr[current_features], y_val)],
            verbose=False,
        )

        # Evaluate Validation MAE
        y_val_pred = model.predict(X_va_curr[current_features])
        val_mae = float(mean_absolute_error(y_val, y_val_pred))
        logger.info("Iteration %d Validation MAE: %.5f", iteration, val_mae)

        # Track global minimum
        if val_mae < best_mae:
            logger.info(
                "New global minimum MAE found: %.5f -> %.5f (features: %d)",
                best_mae,
                val_mae,
                len(current_features),
            )
            best_mae = val_mae
            best_features = list(current_features)
            best_model = model

        # Check stopping criterion: consecutive degradation
        if prev_mae is not None and val_mae > prev_mae:
            consecutive_degradations += 1
            logger.warning(
                "MAE degraded: %.5f > %.5f (consecutive degradations: %d/%d)",
                val_mae,
                prev_mae,
                consecutive_degradations,
                patience,
            )
        else:
            consecutive_degradations = 0

        prev_mae = val_mae

        if consecutive_degradations >= patience:
            logger.info(
                "Stopping criterion met: Validation MAE degraded for %d consecutive iterations.",
                patience,
            )
            history_records.append({
                "iteration": iteration,
                "n_features": len(current_features),
                "val_mae": val_mae,
                "dropped_features": [],
            })
            break

        # Compute SHAP importance strictly on X_val
        shap_ranking = calculate_val_shap_importance(
            model=model,
            X_val=X_va_curr[current_features],
        )

        # Determine number of bottom features to drop (at least 1, leaving at least 1 feature)
        n_to_drop = max(1, int(np.ceil(len(current_features) * drop_fraction)))
        n_to_drop = min(n_to_drop, len(current_features) - 1)

        features_to_drop = shap_ranking.index[:n_to_drop].tolist()
        logger.info(
            "Dropping bottom %d features (lowest val SHAP impact): %s",
            len(features_to_drop),
            features_to_drop,
        )

        history_records.append({
            "iteration": iteration,
            "n_features": len(current_features),
            "val_mae": val_mae,
            "dropped_features": features_to_drop,
        })

        # Update feature list for next iteration
        current_features = [f for f in current_features if f not in features_to_drop]
        iteration += 1

    history_df = pd.DataFrame(history_records)

    # Ensure best model is retrained if not captured or if final loop state differs
    if best_model is None or list(best_model.feature_names_in_) != best_features:
        logger.info("Refitting final model on best feature subset (%d features)...", len(best_features))
        best_model = xgb.XGBRegressor(**default_xgb_params)
        best_model.fit(X_tr_curr[best_features], y_train, verbose=False)

    logger.info(
        "SHAP-RFE complete! Optimal subset: %d features | Global Min Val MAE: %.5f",
        len(best_features),
        best_mae,
    )

    return SHAP_RFE_Result(
        best_features=best_features,
        best_mae=best_mae,
        best_model=best_model,
        history=history_df,
        dropped_collinear_features=dropped_collinear,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Generate synthetic OCPM data mimicking graph summary metrics
    np.random.seed(42)
    n_train, n_val = 1000, 300

    def generate_synthetic_ocpm_data(n: int) -> Tuple[pd.DataFrame, pd.Series]:
        # True informative signals
        p2_elapsed = np.random.exponential(scale=10.0, size=n)
        p5_sync_time = np.random.exponential(scale=5.0, size=n)
        p9_service_time = np.random.normal(loc=15.0, scale=3.0, size=n)
        c5_activity_count = np.random.poisson(lam=4, size=n)

        # Collinear feature: P10 waiting time strongly correlated with P5 sync time (r > 0.9)
        p10_waiting_time = p5_sync_time * 0.95 + np.random.normal(0, 0.2, size=n)

        # Pure noise features
        noise_features = {f"noise_{i}": np.random.normal(0, 1, size=n) for i in range(1, 10)}

        df = pd.DataFrame({
            "P2_elapsed_time": p2_elapsed,
            "P5_sync_time": p5_sync_time,
            "P10_waiting_time": p10_waiting_time,  # Should be pruned by correlation filter
            "P9_service_time": p9_service_time,
            "C5_activity_count": c5_activity_count,
            **noise_features,
        })

        # Remaining time regression target (P3)
        y = 50.0 - 0.8 * p2_elapsed - 1.2 * p5_sync_time + 0.5 * p9_service_time + np.random.normal(0, 1, n)
        return df, pd.Series(y, name="P3_remaining_time")

    X_tr, y_tr = generate_synthetic_ocpm_data(n_train)
    X_va, y_va = generate_synthetic_ocpm_data(n_val)

    # Run the pipeline
    result = run_shap_rfe_pipeline(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        correlation_threshold=0.85,
        drop_fraction=0.15,
        patience=2,
    )

    print("\n" + "=" * 50)
    print("Pipeline Execution Summary:")
    print(f"Collinear Features Dropped: {result.dropped_collinear_features}")
    print(f"Optimal Feature Count:      {len(result.best_features)}")
    print(f"Best Selected Features:     {result.best_features}")
    print(f"Global Minimum Val MAE:     {result.best_mae:.4f}")
    print("=" * 50)
    print("\nIteration History:")
    print(result.history)

