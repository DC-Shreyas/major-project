"""Canonical synthetic property dataset, model evaluation, and interactive estimates.

The supplied synthetic tables are the only modeling inputs for the current
project workflow. Their generated prices support software development and a
synthetic-data study; they are not observed transaction records.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROPERTY_FILE = "synthetic_property_valuation_data.csv"
DEED_FILE = "synthetic_registered_deeds_validation.csv"
TRAIN_QUARTERS = ("2023Q2", "2023Q3")
CALIBRATION_QUARTER = "2023Q4"
EVALUATION_QUARTER = "2024Q1"
CONFORMAL_ALPHA = 0.10
SEED = 2026
QUARTER_ORDER = ("2023Q2", "2023Q3", "2023Q4", "2024Q1")

NUMERIC_FEATURES = [
    "log_area_sqft", "bedrooms", "building_age_years", "floor_number",
    "log_guideline_rate", "log_guidance_value", "log_asking_price",
    "log_declared_value", "asking_to_guidance_log_ratio",
    "declared_to_guidance_log_ratio",
]
CATEGORICAL_FEATURES = ["sro_area", "micro_market_id", "property_type"]


def load_synthetic_records(project_root: Path) -> pd.DataFrame:
    """Load and validate the paired generated property and deed tables."""
    dataset = project_root / "dataset"
    property_path, deed_path = dataset / PROPERTY_FILE, dataset / DEED_FILE
    missing = [str(path) for path in (property_path, deed_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required synthetic dataset file(s) not found: " + ", ".join(missing))
    properties = pd.read_csv(property_path, encoding="utf-8-sig")
    deeds = pd.read_csv(deed_path, encoding="utf-8-sig")
    # The property and deed tables use slightly different names for this shared value.
    if "synthetic_area_derived_guidance_value_inr" not in properties.columns:
        if "synthetic_area_derived_guideline_value_inr" in properties.columns:
            properties["synthetic_area_derived_guidance_value_inr"] = properties[
                "synthetic_area_derived_guideline_value_inr"
            ]
    required_properties = {
        "synthetic_property_id", "sro_area", "micro_market_id", "micro_market",
        "property_type", "synthetic_area_sqft", "bedrooms", "building_age_years",
        "floor_number", "guideline_rate_inr_per_sqm",
        "synthetic_area_derived_guidance_value_inr", "synthetic_latent_true_price_inr",
        "synthetic_asking_price_inr", "synthetic_declared_registration_value_inr",
        "synthetic_quarter",
    }
    required_deeds = {
        "synthetic_property_id", "synthetic_deed_id", "synthetic_registration_date",
        "synthetic_registration_quarter", "validation_sample_flag",
        "synthetic_validation_price_inr", "validation_reference_type",
    }
    missing_properties = sorted(required_properties - set(properties.columns))
    missing_deeds = sorted(required_deeds - set(deeds.columns))
    if missing_properties or missing_deeds:
        raise ValueError(
            f"Synthetic data schema mismatch. Property columns missing: {missing_properties}; "
            f"deed/validation columns missing: {missing_deeds}"
        )
    if properties["synthetic_property_id"].duplicated().any():
        raise ValueError("Property table has duplicate synthetic_property_id values")
    if deeds["synthetic_property_id"].duplicated().any():
        raise ValueError("Deed table has duplicate synthetic_property_id values")
    if set(properties["synthetic_property_id"]) != set(deeds["synthetic_property_id"]):
        raise ValueError("Property and deed tables do not contain the same property IDs")
    linkage = [
        "synthetic_property_id", "synthetic_deed_id", "synthetic_registration_date",
        "synthetic_registration_quarter", "validation_sample_flag",
        "synthetic_validation_price_inr", "validation_reference_type",
    ]
    if "synthetic_underreported_flag" in deeds:
        linkage.append("synthetic_underreported_flag")
    frame = properties.merge(
        deeds[linkage], on="synthetic_property_id", how="inner",
        validate="one_to_one", suffixes=("", "_deed"),
    )
    if not frame["synthetic_quarter"].eq(frame["synthetic_registration_quarter"]).all():
        raise ValueError("Property quarter and deed registration quarter disagree")

    numeric = [
        "synthetic_area_sqft", "bedrooms", "building_age_years", "floor_number",
        "guideline_rate_inr_per_sqm", "synthetic_area_derived_guidance_value_inr",
        "synthetic_latent_true_price_inr", "synthetic_asking_price_inr",
        "synthetic_declared_registration_value_inr", "validation_sample_flag",
        "synthetic_validation_price_inr",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    positive = [
        "synthetic_area_sqft", "guideline_rate_inr_per_sqm",
        "synthetic_area_derived_guidance_value_inr", "synthetic_latent_true_price_inr",
        "synthetic_asking_price_inr", "synthetic_declared_registration_value_inr",
    ]
    if frame[positive].isna().any().any() or frame[positive].le(0).any().any():
        raise ValueError("Synthetic property data contains missing or non-positive model inputs")
    frame["validation_sample_flag"] = frame["validation_sample_flag"].fillna(0).astype(int)
    if not frame["validation_sample_flag"].isin((0, 1)).all():
        raise ValueError("validation_sample_flag must contain only 0 or 1")
    if frame.loc[frame["validation_sample_flag"].eq(1), "synthetic_validation_price_inr"].isna().any():
        raise ValueError("Flagged validation records must contain a synthetic validation price")
    if not frame["synthetic_quarter"].isin(QUARTER_ORDER).all():
        raise ValueError("The supplied property table contains an unrecognized quarter")
    lower = np.maximum(
        frame["synthetic_declared_registration_value_inr"].to_numpy(dtype=float),
        frame["synthetic_area_derived_guidance_value_inr"].to_numpy(dtype=float),
    )
    upper = frame["synthetic_asking_price_inr"].to_numpy(dtype=float)
    target = frame["synthetic_latent_true_price_inr"].to_numpy(dtype=float)
    if (lower > upper).any() or (target < lower).any() or (target > upper).any():
        raise ValueError("Generated prices do not satisfy the declared/guidance to asking-price bounds")
    return frame.reset_index(drop=True)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build consistent model features from either source rows or form values."""
    area = pd.to_numeric(frame["synthetic_area_sqft"], errors="coerce").clip(lower=1)
    guidance = pd.to_numeric(frame["synthetic_area_derived_guidance_value_inr"], errors="coerce").clip(lower=1)
    asking = pd.to_numeric(frame["synthetic_asking_price_inr"], errors="coerce").clip(lower=1)
    declared = pd.to_numeric(frame["synthetic_declared_registration_value_inr"], errors="coerce").clip(lower=1)
    rate = pd.to_numeric(frame["guideline_rate_inr_per_sqm"], errors="coerce").clip(lower=1)
    features = pd.DataFrame(
        {
            "log_area_sqft": np.log(area),
            "bedrooms": pd.to_numeric(frame.get("bedrooms"), errors="coerce"),
            "building_age_years": pd.to_numeric(frame.get("building_age_years"), errors="coerce"),
            "floor_number": pd.to_numeric(frame.get("floor_number"), errors="coerce"),
            "log_guideline_rate": np.log(rate),
            "log_guidance_value": np.log(guidance),
            "log_asking_price": np.log(asking),
            "log_declared_value": np.log(declared),
            "asking_to_guidance_log_ratio": np.log(asking / guidance),
            "declared_to_guidance_log_ratio": np.log(declared / guidance),
            "sro_area": frame.get("sro_area", pd.Series("__UNKNOWN__", index=frame.index)).fillna("__UNKNOWN__").astype(str),
            "micro_market_id": frame.get("micro_market_id", pd.Series("__NEW_MARKET__", index=frame.index)).fillna("__NEW_MARKET__").astype(str),
            "property_type": frame.get("property_type", pd.Series("__UNKNOWN__", index=frame.index)).fillna("__UNKNOWN__").astype(str),
        },
        index=frame.index,
    )
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _make_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
    ])
    features = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", features),
        ("model", RandomForestRegressor(
            n_estimators=260, max_depth=18, min_samples_leaf=4,
            max_features=0.9, random_state=SEED, n_jobs=-1,
        )),
    ])


def _finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.log(1.45)
    index = min(len(values) - 1, max(0, int(math.ceil((len(values) + 1) * (1 - alpha))) - 1))
    return float(np.sort(values)[index])


def _metric_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    absolute = np.abs(actual - predicted)
    return {
        "mae_inr": round(float(mean_absolute_error(actual, predicted)), 2),
        "mape_percent": round(float(np.mean(absolute / np.maximum(actual, 1)) * 100), 3),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


class SyntheticMarketModel:
    """Fit the current synthetic-only model and serve estimates for property inputs."""

    def __init__(self, records: pd.DataFrame):
        self.records = records.copy()
        training = self.records["synthetic_quarter"].isin(("2023Q2", "2023Q3"))
        calibration = (
            self.records["synthetic_quarter"].eq(CALIBRATION_QUARTER)
            & self.records["validation_sample_flag"].eq(1)
        )
        evaluation = self.records["synthetic_quarter"].eq(EVALUATION_QUARTER)
        if not training.any() or calibration.sum() < 20 or not evaluation.any():
            raise ValueError("The generated dataset does not contain enough rows for the documented time split")
        self.pipeline = _make_pipeline()
        self.pipeline.fit(
            _feature_frame(self.records.loc[training]),
            np.log(self.records.loc[training, "synthetic_latent_true_price_inr"].to_numpy(dtype=float)),
        )
        calibration_rows = self.records.loc[calibration]
        calibration_prediction = np.exp(self.pipeline.predict(_feature_frame(calibration_rows)).astype(float))
        calibration_truth = calibration_rows["synthetic_validation_price_inr"].to_numpy(dtype=float)
        calibration_error = np.abs(
            np.log(np.maximum(calibration_truth, 1)) - np.log(np.maximum(calibration_prediction, 1))
        )
        self.log_error_quantile = _finite_sample_quantile(calibration_error, CONFORMAL_ALPHA)

        evaluation_rows = self.records.loc[evaluation]
        raw_prediction = np.exp(self.pipeline.predict(_feature_frame(evaluation_rows)).astype(float))
        lower_bound = np.maximum(
            evaluation_rows["synthetic_declared_registration_value_inr"].to_numpy(dtype=float),
            evaluation_rows["synthetic_area_derived_guidance_value_inr"].to_numpy(dtype=float),
        )
        upper_bound = evaluation_rows["synthetic_asking_price_inr"].to_numpy(dtype=float)
        bounded = np.clip(raw_prediction, lower_bound, upper_bound)
        truth = evaluation_rows["synthetic_latent_true_price_inr"].to_numpy(dtype=float)
        interval_lower = np.maximum(lower_bound, bounded * np.exp(-self.log_error_quantile))
        interval_upper = np.minimum(upper_bound, bounded * np.exp(self.log_error_quantile))
        coverage = float(np.mean((truth >= interval_lower) & (truth <= interval_upper)))
        self.evaluation_predictions = pd.DataFrame({
            "synthetic_property_id": evaluation_rows["synthetic_property_id"].to_numpy(),
            "sro_area": evaluation_rows["sro_area"].to_numpy(),
            "micro_market_id": evaluation_rows["micro_market_id"].to_numpy(),
            "micro_market": evaluation_rows["micro_market"].to_numpy(),
            "property_type": evaluation_rows["property_type"].to_numpy(),
            "quarter": evaluation_rows["synthetic_quarter"].to_numpy(),
            "area_sqft": evaluation_rows["synthetic_area_sqft"].to_numpy(),
            "asking_price_inr": evaluation_rows["synthetic_asking_price_inr"].to_numpy(),
            "declared_price_inr": evaluation_rows["synthetic_declared_registration_value_inr"].to_numpy(),
            "guidance_value_inr": evaluation_rows["synthetic_area_derived_guidance_value_inr"].to_numpy(),
            "estimated_price_inr": np.rint(bounded).astype("int64"),
            "interval_lower_inr": np.rint(interval_lower).astype("int64"),
            "interval_upper_inr": np.rint(interval_upper).astype("int64"),
            "synthetic_reference_price_inr": evaluation_rows["synthetic_latent_true_price_inr"].to_numpy(),
            "validation_sample_flag": evaluation_rows["validation_sample_flag"].to_numpy(),
        })
        baseline = np.sqrt(
            evaluation_rows["synthetic_asking_price_inr"].to_numpy(dtype=float)
            * evaluation_rows["synthetic_declared_registration_value_inr"].to_numpy(dtype=float)
        )
        self.report = {
            "status": "synthetic_dataset_model_ready",
            "data_source": f"dataset/{PROPERTY_FILE} + dataset/{DEED_FILE}",
            "records": int(len(self.records)),
            "distinct_micro_markets": int(self.records["micro_market_id"].nunique()),
            "sro_areas": sorted(self.records["sro_area"].astype(str).unique().tolist()),
            "property_types": sorted(self.records["property_type"].astype(str).unique().tolist()),
            "quarters": list(QUARTER_ORDER),
            "training_quarters": ["2023Q2", "2023Q3"],
            "training_rows": int(training.sum()),
            "calibration_quarter": CALIBRATION_QUARTER,
            "calibration_rows": int(calibration.sum()),
            "evaluation_quarter": EVALUATION_QUARTER,
            "evaluation_rows": int(evaluation.sum()),
            "synthetic_validation_anchors_in_evaluation": int(evaluation_rows["validation_sample_flag"].sum()),
            "model": "Random forest on property, micro-market, guidance, asking, and declared-price features; estimates and intervals are bounded by the supplied guidance, declared, and asking values.",
            "calibration": {
                "nominal_coverage_percent": (1 - CONFORMAL_ALPHA) * 100,
                "observed_coverage_percent": round(coverage * 100, 2),
                "mean_width_inr": round(float(np.mean(interval_upper - interval_lower)), 2),
                "log_error_quantile": round(self.log_error_quantile, 6),
                "reference_rows": int(calibration.sum()),
            },
            "holdout_metrics_against_synthetic_reference_price": _metric_summary(truth, bounded),
            "geometric_signal_baseline_metrics": _metric_summary(truth, baseline),
            "price_summary": {
                "median_reference_price_inr": int(self.records["synthetic_latent_true_price_inr"].median()),
                "median_asking_price_inr": int(self.records["synthetic_asking_price_inr"].median()),
                "median_declared_price_inr": int(self.records["synthetic_declared_registration_value_inr"].median()),
                "median_price_per_sqft_inr": round(float(
                    (self.records["synthetic_latent_true_price_inr"] / self.records["synthetic_area_sqft"]).median()
                ), 2),
            },
            "scope_note": (
                "Generated files are the canonical dataset for this software project. All model metrics describe "
                "recovery of generated synthetic reference prices and do not measure real-world accuracy."
            ),
        }

    def estimate(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return a bounded estimate and earlier-quarter synthetic comparables."""
        try:
            area = float(values["area_sqft"])
            bedrooms = float(values["bedrooms"]) if values.get("bedrooms") not in (None, "") else np.nan
            age = float(values["building_age_years"]) if values.get("building_age_years") not in (None, "") else np.nan
            floor = float(values["floor_number"]) if values.get("floor_number") not in (None, "") else np.nan
            rate = float(values["guideline_rate_inr_per_sqm"])
            asking = float(values["asking_price_inr"])
            declared = float(values["declared_price_inr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Enter valid numeric values for area, guidance, asking, and declared price.") from exc
        if area <= 0 or rate <= 0 or asking <= 0 or declared <= 0:
            raise ValueError("Area, guidance rate, asking price, and declared price must be greater than zero.")
        if not np.isfinite([area, rate, asking, declared]).all():
            raise ValueError("The entered values must be finite numbers.")
        if not np.isnan(bedrooms) and bedrooms < 0:
            raise ValueError("Bedrooms cannot be negative.")
        if not np.isnan(age) and age < 0:
            raise ValueError("Building age cannot be negative.")
        if not np.isnan(floor) and floor < 0:
            raise ValueError("Floor cannot be negative.")

        sro = str(values.get("sro_area") or "")
        market_id = str(values.get("micro_market_id") or "__NEW_MARKET__")
        property_type = str(values.get("property_type") or "")
        if sro not in self.report["sro_areas"]:
            raise ValueError("Choose one of the three SRO areas in the dataset.")
        if property_type not in self.report["property_types"]:
            raise ValueError("Choose a property type listed in the dataset.")
        quarter = str(values.get("as_of_quarter") or EVALUATION_QUARTER)
        if quarter not in self.report["quarters"]:
            raise ValueError("Choose a quarter covered by the dataset.")

        guidance_value = rate * (area / 10.7639)
        effective_lower = max(declared, guidance_value)
        if effective_lower > asking:
            raise ValueError("The higher of the guidance reference and declared value must not exceed the asking price.")
        model_row = pd.DataFrame([{
            "synthetic_area_sqft": area,
            "bedrooms": bedrooms,
            "building_age_years": age,
            "floor_number": floor,
            "guideline_rate_inr_per_sqm": rate,
            "synthetic_area_derived_guidance_value_inr": guidance_value,
            "synthetic_asking_price_inr": asking,
            "synthetic_declared_registration_value_inr": declared,
            "sro_area": sro,
            "micro_market_id": market_id,
            "property_type": property_type,
        }])
        raw = float(np.exp(self.pipeline.predict(_feature_frame(model_row))[0]))
        estimate = float(np.clip(raw, effective_lower, asking))
        lower = float(max(effective_lower, estimate * math.exp(-self.log_error_quantile)))
        upper = float(min(asking, estimate * math.exp(self.log_error_quantile)))
        comparables = self._comparables(
            area=area, bedrooms=bedrooms, age=age, sro=sro,
            market_id=market_id, property_type=property_type, quarter=quarter,
        )
        return {
            "estimated_price_inr": int(round(estimate)),
            "interval_lower_inr": int(round(lower)),
            "interval_upper_inr": int(round(upper)),
            "price_per_sqft_inr": round(estimate / area, 2),
            "guidance_value_inr": int(round(guidance_value)),
            "declared_price_inr": int(round(declared)),
            "asking_price_inr": int(round(asking)),
            "bound_position_percent": round(float(
                (estimate - effective_lower) / max(asking - effective_lower, 1) * 100
            ), 1),
            "calibration": {
                "nominal_coverage_percent": (1 - CONFORMAL_ALPHA) * 100,
                "observed_coverage_percent": self.report["calibration"]["observed_coverage_percent"],
                "interval_reference_rows": self.report["calibration"]["reference_rows"],
                "reference_quarter": CALIBRATION_QUARTER,
            },
            "comparables": comparables,
            "scope_note": "Estimate learned from generated synthetic records; it is not a real-world appraisal.",
        }

    def _comparables(
        self, *, area: float, bedrooms: float, age: float, sro: str,
        market_id: str, property_type: str, quarter: str,
    ) -> list[dict[str, Any]]:
        quarter_index = QUARTER_ORDER.index(quarter)
        rows = self.records.loc[
            self.records["synthetic_quarter"].map(
                {value: index for index, value in enumerate(QUARTER_ORDER)}
            ).lt(quarter_index)
            & self.records["synthetic_area_sqft"].gt(0)
        ].copy()
        exact_group = rows.loc[rows["sro_area"].eq(sro) & rows["property_type"].eq(property_type)]
        if len(exact_group) >= 5:
            rows = exact_group
        else:
            same_sro = rows.loc[rows["sro_area"].eq(sro)]
            rows = same_sro if len(same_sro) >= 5 else rows
        area_distance = np.abs(np.log(rows["synthetic_area_sqft"].to_numpy(dtype=float) / max(area, 1)))
        bedroom_values = rows["bedrooms"].to_numpy(dtype=float)
        bedroom_distance = np.zeros(len(rows), dtype=float)
        if np.isfinite(bedrooms):
            bedroom_distance = np.abs(np.nan_to_num(bedroom_values, nan=bedrooms) - bedrooms) * 0.18
        age_values = rows["building_age_years"].to_numpy(dtype=float)
        age_distance = np.zeros(len(rows), dtype=float)
        if np.isfinite(age):
            age_distance = np.abs(np.nan_to_num(age_values, nan=age) - age) / 100
        market_penalty = np.where(rows["micro_market_id"].to_numpy() == market_id, 0.0, 0.12)
        type_penalty = np.where(rows["property_type"].to_numpy() == property_type, 0.0, 0.4)
        distance = area_distance + bedroom_distance + age_distance + market_penalty + type_penalty
        rows = rows.assign(_distance=distance).sort_values("_distance").head(5)
        return [
            {
                "micro_market": str(row["micro_market"]),
                "property_type": str(row["property_type"]),
                "quarter": str(row["synthetic_quarter"]),
                "area_sqft": int(row["synthetic_area_sqft"]),
                "asking_price_inr": int(row["synthetic_asking_price_inr"]),
                "declared_price_inr": int(row["synthetic_declared_registration_value_inr"]),
                "similarity_percent": max(0, min(100, round(math.exp(-float(row["_distance"])) * 100))),
            }
            for _, row in rows.iterrows()
        ]


def _markdown_report(report: dict) -> str:
    holdout = report["holdout_metrics_against_synthetic_reference_price"]
    baseline = report["geometric_signal_baseline_metrics"]
    calibration = report["calibration"]
    return "\n".join([
        "# Synthetic property model report",
        "",
        f"**Status:** {report['status']}",
        "",
        f"This workflow uses **{report['records']:,} paired generated property/deed records** as the project’s canonical dataset. All prices and labels are synthetic. Results describe performance on generated records only and are not evidence of real Bengaluru market accuracy.",
        "",
        "## Data and split",
        "",
        f"- Training: {', '.join(report['training_quarters'])} ({report['training_rows']:,} rows).",
        f"- Interval calibration: {report['calibration_quarter']} ({report['calibration_rows']:,} flagged generated reference prices).",
        f"- Holdout: {report['evaluation_quarter']} ({report['evaluation_rows']:,} rows).",
        f"- Holdout coverage: {calibration['observed_coverage_percent']:.1f}% for nominal {calibration['nominal_coverage_percent']:.0f}% intervals; mean width INR {calibration['mean_width_inr']:,.0f}.",
        "",
        "## Holdout results against generated reference prices",
        "",
        "| Method | MAE (INR) | MAPE | R² |",
        "|---|---:|---:|---:|",
        f"| Random forest with bounded price signals | {holdout['mae_inr']:,.0f} | {holdout['mape_percent']:.2f}% | {holdout['r2']:.3f} |",
        f"| Geometric mean of asking and declared signals | {baseline['mae_inr']:,.0f} | {baseline['mape_percent']:.2f}% | {baseline['r2']:.3f} |",
        "",
        "## Interpretation",
        "",
        "Training targets and interval calibration references are generated values. Estimates are constrained by the guidance reference, declared amount, and asking amount. Read this as a reproducible software demonstration on this dataset, not as a bank, surveyor, or market valuation.",
        "",
    ])


def run_synthetic_market_pipeline(project_root: Path) -> dict:
    """Build the canonical synthetic-only table, evaluation, and model report."""
    records = load_synthetic_records(project_root)
    model = SyntheticMarketModel(records)
    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    records.to_csv(processed / "property_records.csv", index=False, encoding="utf-8-sig")
    model.evaluation_predictions.to_csv(
        processed / "market_model_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = model.report
    (reports / "market_model_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports / "market_model_metrics.md").write_text(_markdown_report(report), encoding="utf-8")
    return report
