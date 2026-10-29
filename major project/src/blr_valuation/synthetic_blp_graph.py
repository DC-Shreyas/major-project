"""Simulation-only BLP-Graph prototype using the supplied synthetic fixtures.

The latent-price columns in these fixtures are generated values, not observed
transactions. This module exercises the architecture and never promotes its
metrics to empirical Bengaluru results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import SplineTransformer, StandardScaler
from torch import nn
from torch.nn import functional as F


SEED = 2026
TRAIN_QUARTERS = ["2023Q2", "2023Q3"]
CALIBRATION_QUARTER = "2023Q4"
TEST_QUARTER = "2024Q1"
CONFORMAL_ALPHA = 0.10
OOV_MARKET_ID = "__BLP_GRAPH_OOV_MARKET__"


def _load_synthetic(project_root: Path) -> pd.DataFrame:
    properties = pd.read_csv(
        project_root / "dataset" / "synthetic_property_valuation_data.csv",
        encoding="utf-8-sig",
    )
    if "synthetic_area_derived_guidance_value_inr" not in properties.columns:
        if "synthetic_area_derived_guideline_value_inr" in properties.columns:
            properties["synthetic_area_derived_guidance_value_inr"] = properties[
                "synthetic_area_derived_guideline_value_inr"
            ]
    deeds = pd.read_csv(
        project_root / "dataset" / "synthetic_registered_deeds_validation.csv",
        encoding="utf-8-sig",
    )
    validation = deeds[
        [
            "synthetic_property_id",
            "validation_sample_flag",
            "synthetic_validation_price_inr",
            "validation_reference_type",
        ]
    ]
    frame = properties.merge(validation, on="synthetic_property_id", how="left", validate="one_to_one")
    if frame["validation_sample_flag"].isna().any():
        raise ValueError("Synthetic deed fixture is missing property IDs from the synthetic property table")

    numeric = [
        "synthetic_area_sqft",
        "bedrooms",
        "building_age_years",
        "floor_number",
        "guideline_rate_inr_per_sqm",
        "synthetic_area_derived_guideline_value_inr",
        "synthetic_latent_true_price_inr",
        "synthetic_asking_price_inr",
        "synthetic_declared_registration_value_inr",
        "synthetic_validation_price_inr",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["validation_sample_flag"] = frame["validation_sample_flag"].fillna(0).astype(int)
    frame["quarter_index"] = pd.Categorical(
        frame["synthetic_quarter"],
        categories=["2023Q2", "2023Q3", "2023Q4", "2024Q1"],
        ordered=True,
    ).codes
    frame["market_id"] = frame["micro_market_id"].astype(str)
    frame["market_name"] = frame["micro_market"].astype(str)
    if frame["quarter_index"].lt(0).any():
        raise ValueError("Synthetic records include a quarter outside the documented simulation period")
    if not (frame["synthetic_declared_registration_value_inr"] <= frame["synthetic_asking_price_inr"]).all():
        raise ValueError("Synthetic fixture contains records outside the proposal's declared-price/asking-price bounds")
    if not (
        frame["synthetic_declared_registration_value_inr"]
        >= frame["synthetic_area_derived_guideline_value_inr"]
    ).all():
        raise ValueError("Synthetic fixture contains declared values below its guidance reference")
    return frame


class _GAMFeatureEncoder:
    """Spline basis plus one-hot terms for an additive hedonic GAM head."""

    numeric_columns = ("log_area_sqft", "bedrooms", "building_age_years", "floor_number")

    def __init__(self):
        self.medians: pd.Series | None = None
        self.spline: SplineTransformer | None = None
        self.categories: dict[str, list[str]] = {}
        self.dummy_columns: list[str] = []
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.feature_groups: dict[str, list[int]] = {}

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "log_area_sqft": np.log(pd.to_numeric(frame["synthetic_area_sqft"], errors="coerce").clip(lower=1)),
                "bedrooms": pd.to_numeric(frame["bedrooms"], errors="coerce"),
                "building_age_years": pd.to_numeric(frame["building_age_years"], errors="coerce"),
                "floor_number": pd.to_numeric(frame["floor_number"], errors="coerce"),
            },
            index=frame.index,
        ).replace([np.inf, -np.inf], np.nan)

    def _categorical(self, frame: pd.DataFrame) -> pd.DataFrame:
        categories = frame[["property_type", "sro_area"]].fillna("__MISSING__").astype(str)
        dummies = pd.get_dummies(categories, columns=["property_type", "sro_area"], dtype=float)
        return dummies.reindex(columns=self.dummy_columns, fill_value=0.0)

    def fit_transform(self, frame: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
        numeric = self._numeric(frame)
        self.medians = numeric.loc[train_mask].median().fillna(0.0)
        filled = numeric.fillna(self.medians).fillna(0.0)
        self.spline = SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="linear")
        self.spline.fit(filled.loc[train_mask])
        self.categories = {
            column: sorted(frame.loc[train_mask, column].fillna("__MISSING__").astype(str).unique().tolist())
            for column in ("property_type", "sro_area")
        }
        train_categories = frame.loc[train_mask, ["property_type", "sro_area"]].fillna("__MISSING__").astype(str)
        train_dummies = pd.get_dummies(
            train_categories, columns=["property_type", "sro_area"], dtype=float
        )
        self.dummy_columns = list(train_dummies.columns)
        spline_names = list(self.spline.get_feature_names_out(list(self.numeric_columns)))
        self.feature_names = spline_names + self.dummy_columns
        self.feature_groups = {}
        for column in self.numeric_columns:
            self.feature_groups[column] = [i for i, name in enumerate(spline_names) if name.startswith(column + "_")]
        offset = len(spline_names)
        self.feature_groups["property_type"] = [offset + i for i, name in enumerate(self.dummy_columns) if name.startswith("property_type_")]
        self.feature_groups["sro_area"] = [offset + i for i, name in enumerate(self.dummy_columns) if name.startswith("sro_area_")]
        raw = np.column_stack([self.spline.transform(filled), self._categorical(frame).to_numpy(dtype=float)])
        self.scaler = StandardScaler().fit(raw[train_mask])
        return self.scaler.transform(raw).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.spline is None or self.scaler is None:
            raise RuntimeError("GAM feature encoder has not been fitted")
        filled = self._numeric(frame).fillna(self.medians).fillna(0.0)
        raw = np.column_stack([self.spline.transform(filled), self._categorical(frame).to_numpy(dtype=float)])
        return self.scaler.transform(raw).astype(np.float32)


def _prepare_features(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict[str, int], _GAMFeatureEncoder]:
    """Fit feature transforms and graph profiles on training rows only."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(frame),) or not train_mask.any():
        raise ValueError("Feature preparation requires a non-empty training mask aligned to the frame")
    encoder = _GAMFeatureEncoder()
    unit_features = encoder.fit_transform(frame, train_mask)
    known_market_ids = sorted(frame.loc[train_mask, "market_id"].astype(str).unique().tolist())
    if OOV_MARKET_ID in known_market_ids:
        raise ValueError(f"Reserved synthetic market ID is present in the input: {OOV_MARKET_ID}")
    market_ids = [*known_market_ids, OOV_MARKET_ID]
    market_to_index = {market_id: index for index, market_id in enumerate(market_ids)}
    oov_index = market_to_index[OOV_MARKET_ID]
    market_indices = frame["market_id"].astype(str).map(market_to_index).fillna(oov_index).to_numpy(dtype=np.int64)

    profile_numeric = encoder._numeric(frame).fillna(encoder.medians).fillna(0.0)
    raw_profiles = []
    profile_names: list[str] = []
    def safe_mean(values: pd.Series) -> float:
        result = values.mean()
        return float(result) if pd.notna(result) else 0.0

    training_sros = sorted(frame.loc[train_mask, "sro_area"].astype(str).unique().tolist())
    for market_id in known_market_ids:
        group_mask = train_mask & frame["market_id"].astype(str).eq(market_id).to_numpy()
        group = frame.loc[group_mask]
        features = {
            "mean_log_area": safe_mean(profile_numeric.loc[group.index, "log_area_sqft"]),
            "mean_bedrooms": safe_mean(group["bedrooms"]),
            "mean_building_age": safe_mean(group["building_age_years"]),
            "share_site": float(group["property_type"].eq("Residential Site").mean()),
            "share_apartment": float(group["property_type"].eq("Apartment/Flat").mean()),
            "share_villa": float(group["property_type"].eq("Villa/Row House").mean()),
        }
        for sro in training_sros:
            features[f"sro_{sro}"] = float(group["sro_area"].astype(str).eq(sro).mean())
        if not profile_names:
            profile_names = list(features)
        raw_profiles.append([features[name] for name in profile_names])
    raw_profile_array = np.asarray(raw_profiles, dtype=np.float32)
    raw_profile_array = np.nan_to_num(raw_profile_array, nan=0.0, posinf=0.0, neginf=0.0)
    profile_scaler = StandardScaler().fit(raw_profile_array)
    known_profiles = profile_scaler.transform(raw_profile_array).astype(np.float32)
    # A single shared unknown-market node represents cold-start markets. Its zero
    # profile is the training-market mean after standardization.
    profile_array = np.vstack([known_profiles, np.zeros((1, known_profiles.shape[1]), dtype=np.float32)])
    profile_array = np.nan_to_num(profile_array, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        unit_features.astype(np.float32),
        list(encoder.feature_names),
        profile_array,
        market_ids,
        market_to_index,
        encoder,
    )


def _build_market_graph(profiles: np.ndarray, market_ids: list[str], k: int = 5) -> tuple[np.ndarray, list[dict]]:
    squared_norms = np.sum(profiles**2, axis=1, keepdims=True)
    squared_distance = np.maximum(squared_norms + squared_norms.T - 2 * (profiles @ profiles.T), 0.0)
    nonzero = squared_distance[squared_distance > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
    similarity = np.exp(-squared_distance / max(2.0 * bandwidth, 1e-8))
    np.fill_diagonal(similarity, 0.0)
    edge_weights: dict[tuple[int, int], float] = {}
    for source in range(len(market_ids)):
        neighbors = np.argpartition(similarity[source], -min(k, len(market_ids) - 1))[-min(k, len(market_ids) - 1):]
        for target in neighbors:
            if source == int(target):
                continue
            left, right = sorted((source, int(target)))
            edge_weights[(left, right)] = max(
                edge_weights.get((left, right), 0.0),
                float(max(0.0, similarity[source, target])),
            )
    adjacency = np.zeros((len(market_ids), len(market_ids)), dtype=np.float32)
    edges = []
    for (left, right), weight in sorted(edge_weights.items()):
        if weight <= 0:
            continue
        adjacency[left, right] = weight
        adjacency[right, left] = weight
        edges.append(
            {
                "source_market_id": market_ids[left],
                "target_market_id": market_ids[right],
                "similarity_weight": round(weight, 6),
                "edge_type": "functional_covariate_similarity",
            }
        )
    adjacency += np.eye(len(market_ids), dtype=np.float32)
    degree = adjacency.sum(axis=1).clip(min=1e-8)
    inv_sqrt = degree ** -0.5
    normalized_adjacency = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return normalized_adjacency.astype(np.float32), edges


def _build_quarter_history(
    frame: pd.DataFrame,
    market_to_index: dict[str, int],
    use_asking: bool = True,
    use_deeds: bool = True,
    excluded_markets: set[str] | None = None,
) -> np.ndarray:
    quarters = ["2023Q2", "2023Q3", "2023Q4", "2024Q1"]
    history = np.zeros((len(market_to_index), len(quarters), 3), dtype=np.float32)
    oov_index = market_to_index[OOV_MARKET_ID]
    history_values: dict[tuple[int, int], list[list[float]]] = {}
    for market_id, group in frame.groupby("market_id", sort=False):
        market_id = str(market_id)
        if excluded_markets and market_id in excluded_markets:
            continue
        market_index = market_to_index.get(market_id, oov_index)
        for quarter_index, quarter in enumerate(quarters):
            rows = group.loc[group["synthetic_quarter"].eq(quarter)]
            if rows.empty:
                continue
            guidance = rows["synthetic_area_derived_guideline_value_inr"].clip(lower=1)
            ask_ratio = (
                np.log(rows["synthetic_asking_price_inr"].clip(lower=1) / guidance)
                if use_asking else pd.Series(0.0, index=rows.index)
            )
            deed_ratio = (
                np.log(rows["synthetic_declared_registration_value_inr"].clip(lower=1) / guidance)
                if use_deeds else pd.Series(0.0, index=rows.index)
            )
            history_values.setdefault((market_index, quarter_index), []).append([
                float(ask_ratio.median()),
                float(deed_ratio.median()),
                float(np.log1p(len(rows))),
            ])
    for (market_index, quarter_index), values in history_values.items():
        history[market_index, quarter_index, :] = np.median(np.asarray(values, dtype=np.float32), axis=0)
    # Fit scaling on training-quarter histories only; later periods remain out of
    # the feature normalization used by chronological evaluation.
    scaler = StandardScaler()
    scaler.fit(history[:, :2, :].reshape(-1, history.shape[-1]))
    history = scaler.transform(history.reshape(-1, history.shape[-1])).reshape(history.shape).astype(np.float32)
    return history


class _MultiHeadMarketAttention(nn.Module):
    """Attention-weighted message passing over functional micro-market edges."""

    def __init__(self, hidden: int, heads: int = 3):
        super().__init__()
        if hidden % heads:
            raise ValueError("Hidden size must divide evenly across attention heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        count = nodes.shape[0]
        query = self.query(nodes).view(count, self.heads, self.head_dim)
        key = self.key(nodes).view(count, self.heads, self.head_dim)
        value = self.value(nodes).view(count, self.heads, self.head_dim)
        scores = torch.einsum("ihd,jhd->hij", query, key) / math.sqrt(self.head_dim)
        edge_mask = adjacency > 0
        scores = scores + torch.log(adjacency.clamp_min(1e-8))[None, :, :]
        scores = scores.masked_fill(~edge_mask[None, :, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        messages = torch.einsum("hij,jhd->ihd", weights, value).reshape(count, -1)
        return self.output(messages)


class _BoundedLatentGraphNet(nn.Module):
    def __init__(
        self,
        unit_dim: int,
        profile_dim: int,
        market_count: int,
        initial_log_price: float,
        use_graph: bool = True,
        use_temporal: bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        hidden = 12
        temporal_hidden = 8
        self.market_embedding = nn.Embedding(market_count, hidden)
        self.profile_projection = nn.Linear(profile_dim, hidden)
        self.graph_attention = _MultiHeadMarketAttention(hidden, heads=3)
        self.graph_projection = nn.Linear(hidden, hidden)
        self.gru = nn.GRU(input_size=3, hidden_size=temporal_hidden, batch_first=True)
        self.context_projection = nn.Linear(hidden + temporal_hidden, hidden)
        # Linear weights over per-feature spline bases form an additive GAM.
        self.hedonic_head = nn.Linear(unit_dim, 1)
        self.locality_head = nn.Linear(hidden, 1, bias=False)
        self.dual_signal_head = nn.Linear(unit_dim + hidden, 2)
        self.log_sigma = nn.Parameter(torch.tensor(-1.2, dtype=torch.float32))
        nn.init.zeros_(self.hedonic_head.weight)
        nn.init.constant_(self.hedonic_head.bias, initial_log_price)
        nn.init.zeros_(self.locality_head.weight)

    def forward(self, x, market_profiles, adjacency, history, market_indices, quarter_indices):
        market_seed = torch.tanh(self.profile_projection(market_profiles) + self.market_embedding.weight)
        neighbor_state = self.graph_attention(market_seed, adjacency) if self.use_graph else market_seed
        graph_state = torch.tanh(market_seed + self.graph_projection(neighbor_state))
        temporal_by_quarter = []
        for quarter in range(history.shape[1]):
            if quarter == 0 or not self.use_temporal:
                state = torch.zeros(
                    history.shape[0], self.gru.hidden_size, device=history.device, dtype=history.dtype
                )
            else:
                _, hidden = self.gru(history[:, :quarter, :])
                state = hidden[-1]
            temporal_by_quarter.append(state)
        temporal = torch.stack(temporal_by_quarter, dim=1)
        context = torch.tanh(self.context_projection(torch.cat([graph_state[:, None, :].expand(-1, history.shape[1], -1), temporal], dim=-1)))
        row_context = context[market_indices, quarter_indices]
        hedonic_component = self.hedonic_head(x).squeeze(-1)
        locality_component = self.locality_head(row_context).squeeze(-1)
        mean_log_price = hedonic_component + locality_component
        signal_parameters = self.dual_signal_head(torch.cat([x, row_context], dim=-1))
        asking_markup = F.softplus(signal_parameters[:, 0])
        underdeclaration_probability = torch.sigmoid(signal_parameters[:, 1])
        sigma = F.softplus(self.log_sigma) + 0.025
        return (
            mean_log_price, sigma, hedonic_component, locality_component, graph_state,
            asking_markup, underdeclaration_probability,
        )


def _interval_nll(mu: torch.Tensor, sigma: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    z_lower = (lower - mu) / sigma
    if bool(torch.isinf(upper).all()):
        # When asking is ablated, each row is right-censored above its lower
        # bound. The direct survival likelihood avoids undefined gradients at +inf.
        return -torch.special.log_ndtr(-z_lower).mean()
    z_upper = (upper - mu) / sigma
    log_cdf_upper = torch.special.log_ndtr(z_upper)
    log_cdf_lower = torch.special.log_ndtr(z_lower)
    cdf_ratio = torch.exp((log_cdf_lower - log_cdf_upper).clamp(max=-1e-7)).clamp(max=1.0 - 1e-7)
    cdf_log_probability = log_cdf_upper + torch.log1p(-cdf_ratio)
    log_sf_lower = torch.special.log_ndtr(-z_lower)
    log_sf_upper = torch.special.log_ndtr(-z_upper)
    sf_ratio = torch.exp((log_sf_upper - log_sf_lower).clamp(max=-1e-7)).clamp(max=1.0 - 1e-7)
    sf_log_probability = log_sf_lower + torch.log1p(-sf_ratio)
    log_probability = torch.where(z_lower > 0, sf_log_probability, cdf_log_probability)
    return -log_probability.mean()


def _posterior_price(mean_log: np.ndarray, sigma: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    sigma = float(max(sigma, 0.025))
    z_lower = (np.log(lower) - mean_log) / sigma
    z_upper = (np.log(upper) - mean_log) / sigma
    denominator = np.maximum(ndtr(z_upper) - ndtr(z_lower), 1e-12)
    numerator = np.maximum(ndtr(z_upper - sigma) - ndtr(z_lower - sigma), 0.0)
    posterior = np.exp(np.clip(mean_log + 0.5 * sigma**2, -20, 30)) * numerator / denominator
    fallback = np.exp(np.clip(mean_log, -20, 30))
    posterior = np.where(np.isfinite(posterior) & (posterior > 0), posterior, fallback)
    return np.clip(posterior, lower, upper)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    absolute = np.abs(predicted - actual)
    ape = absolute / np.maximum(actual, 1.0)
    return {
        "n": int(len(actual)),
        "mae_inr": round(float(absolute.mean()), 2),
        "rmse_inr": round(float(np.sqrt(np.mean((predicted - actual) ** 2))), 2),
        "mape_percent": round(float(ape.mean() * 100), 3),
        "median_ape_percent": round(float(np.median(ape) * 100), 3),
        "hit_rate_within_10_percent": round(float((ape <= 0.10).mean() * 100), 3),
        "hit_rate_within_20_percent": round(float((ape <= 0.20).mean() * 100), 3),
    }


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    if len(scores) == 0:
        return float("nan")
    ordered = np.sort(scores)
    rank = min(len(ordered), int(np.ceil((len(ordered) + 1) * (1 - alpha))))
    return float(ordered[rank - 1])


def _fit_graph_model(
    frame: pd.DataFrame,
    project_root: Path,
    seed: int = SEED,
    use_graph: bool = True,
    use_temporal: bool = True,
    use_censoring: bool = True,
    use_asking: bool = True,
    use_deeds: bool = True,
    excluded_markets: set[str] | None = None,
    return_predictor: bool = False,
    write_outputs: bool = True,
) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    excluded_markets = {str(market_id) for market_id in (excluded_markets or set())}
    train_mask = frame["synthetic_quarter"].isin(TRAIN_QUARTERS).to_numpy(dtype=bool, copy=True)
    calibration_mask = frame["synthetic_quarter"].eq(CALIBRATION_QUARTER).to_numpy(dtype=bool, copy=True)
    test_mask = frame["synthetic_quarter"].eq(TEST_QUARTER).to_numpy(dtype=bool, copy=True)
    if excluded_markets:
        market_is_seen = ~frame["market_id"].astype(str).isin(excluded_markets).to_numpy()
        train_mask &= market_is_seen
        calibration_mask &= market_is_seen
        test_mask &= ~market_is_seen
    if not train_mask.any() or not calibration_mask.any() or not test_mask.any():
        raise ValueError("Synthetic fixture must contain training, calibration, and test quarters")

    # Fit transforms and construct graph profiles from training rows only.
    x, feature_names, profiles, market_ids, market_to_index, unit_encoder = _prepare_features(frame, train_mask)
    adjacency, edges = _build_market_graph(profiles, market_ids)
    history = _build_quarter_history(
        frame, market_to_index, use_asking=use_asking, use_deeds=use_deeds,
        excluded_markets=excluded_markets,
    )
    oov_index = market_to_index[OOV_MARKET_ID]
    market_indices = frame["market_id"].astype(str).map(market_to_index).fillna(oov_index).to_numpy(dtype=np.int64)
    quarter_indices = frame["quarter_index"].to_numpy(dtype=np.int64)
    asks = frame["synthetic_asking_price_inr"].to_numpy(dtype=np.float32)
    deeds = frame["synthetic_declared_registration_value_inr"].to_numpy(dtype=np.float32)
    guidance = frame["synthetic_area_derived_guideline_value_inr"].to_numpy(dtype=np.float32)
    lower_prices = np.maximum(deeds if use_deeds else guidance, guidance)
    upper_prices = asks if use_asking else np.full_like(asks, np.inf)
    log_lower = torch.tensor(np.log(lower_prices.clip(min=1)), dtype=torch.float32)
    log_upper = torch.tensor(np.log(upper_prices.clip(min=1)), dtype=torch.float32)
    y_true = frame["synthetic_latent_true_price_inr"].to_numpy(dtype=np.float32)
    log_truth = torch.tensor(np.log(y_true.clip(min=1)), dtype=torch.float32)
    asking_markup_target = torch.tensor(
        np.maximum(asks / y_true - 1.0, 0.0), dtype=torch.float32
    )
    underdeclaration_target = torch.tensor((deeds < y_true).astype(np.float32), dtype=torch.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    profile_tensor = torch.tensor(profiles, dtype=torch.float32)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)
    history_tensor = torch.tensor(history, dtype=torch.float32)
    market_tensor = torch.tensor(market_indices, dtype=torch.long)
    quarter_tensor = torch.tensor(quarter_indices, dtype=torch.long)

    anchor_mask = train_mask & frame["validation_sample_flag"].eq(1).to_numpy()
    if anchor_mask.sum() < 20:
        raise ValueError("At least 20 simulated validation anchors are needed to run the demonstration")

    log_lower_values = np.log(lower_prices[train_mask].clip(min=1))
    log_upper_values = np.log(upper_prices[train_mask].clip(min=1))
    interval_midpoint = 0.5 * (log_lower_values + log_upper_values)
    interval_midpoint = interval_midpoint[np.isfinite(interval_midpoint)]
    if not len(interval_midpoint):
        # Asking-only ablations have an infinite upper censoring bound.
        interval_midpoint = log_lower_values[np.isfinite(log_lower_values)]
    initial_log_price = float(np.median(interval_midpoint))
    model = _BoundedLatentGraphNet(
        unit_dim=x.shape[1],
        profile_dim=profiles.shape[1],
        market_count=len(market_ids),
        initial_log_price=initial_log_price,
        use_graph=use_graph,
        use_temporal=use_temporal,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012, weight_decay=2e-4)
    train_tensor = torch.tensor(train_mask, dtype=torch.bool)
    anchor_tensor = torch.tensor(anchor_mask, dtype=torch.bool)
    edge_pairs = [(market_to_index[e["source_market_id"]], market_to_index[e["target_market_id"]]) for e in edges]
    if edge_pairs:
        edge_src = torch.tensor([pair[0] for pair in edge_pairs], dtype=torch.long)
        edge_dst = torch.tensor([pair[1] for pair in edge_pairs], dtype=torch.long)
    else:
        edge_src = edge_dst = torch.tensor([], dtype=torch.long)

    loss_history = []
    for epoch in range(120):
        model.train()
        optimizer.zero_grad()
        mu, sigma, _, _, graph_state, asking_markup, underdeclaration_probability = model(
            x_tensor, profile_tensor, adjacency_tensor, history_tensor, market_tensor, quarter_tensor
        )
        censored_loss = (
            _interval_nll(mu[train_tensor], sigma, log_lower[train_tensor], log_upper[train_tensor])
            if use_censoring else torch.tensor(0.0, dtype=torch.float32)
        )
        anchor_loss = F.mse_loss(mu[anchor_tensor], log_truth[anchor_tensor])
        asking_markup_loss = (
            F.smooth_l1_loss(asking_markup[anchor_tensor], asking_markup_target[anchor_tensor])
            if use_asking else torch.tensor(0.0, dtype=torch.float32)
        )
        underdeclaration_loss = (
            F.binary_cross_entropy(
                underdeclaration_probability[anchor_tensor], underdeclaration_target[anchor_tensor]
            )
            if use_deeds else torch.tensor(0.0, dtype=torch.float32)
        )
        if edge_src.numel() and use_graph:
            graph_loss = (graph_state[edge_src] - graph_state[edge_dst]).pow(2).sum(dim=1).mean()
        else:
            graph_loss = torch.tensor(0.0)
        loss = (
            censored_loss + 14.0 * anchor_loss + 0.002 * graph_loss
            + 0.35 * asking_markup_loss + 0.20 * underdeclaration_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % 10 == 0 or epoch == 119:
            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach()),
                    "interval_nll": float(censored_loss.detach()),
                    "anchor_mse_log_price": float(anchor_loss.detach()),
                    "asking_markup_anchor_loss": float(asking_markup_loss.detach()),
                    "underdeclaration_anchor_loss": float(underdeclaration_loss.detach()),
                    "graph_smoothness": float(graph_loss.detach()),
                }
            )

    model.eval()
    with torch.no_grad():
        mu, sigma_tensor, hedonic_log, locality_log, _, asking_markup_tensor, underdeclaration_tensor = model(
            x_tensor, profile_tensor, adjacency_tensor, history_tensor, market_tensor, quarter_tensor
        )
        mu_array = mu.cpu().numpy()
        hedonic_array = hedonic_log.cpu().numpy()
        locality_array = locality_log.cpu().numpy()
        asking_markup_array = asking_markup_tensor.cpu().numpy()
        underdeclaration_probability_array = underdeclaration_tensor.cpu().numpy()
        sigma = float(sigma_tensor.cpu())

    point = _posterior_price(mu_array, sigma, lower_prices.astype(float), upper_prices.astype(float))
    calibration_anchor_mask = calibration_mask & frame["validation_sample_flag"].eq(1).to_numpy()
    calibration_actual = frame.loc[calibration_anchor_mask, "synthetic_validation_price_inr"].to_numpy(dtype=float)
    calibration_estimated = point[calibration_anchor_mask]
    calibration_error = np.abs(
        np.log(calibration_actual.clip(min=1)) - np.log(calibration_estimated.clip(min=1))
    )
    n_market_clusters = min(6, len(market_ids))
    market_cluster_labels = KMeans(n_clusters=n_market_clusters, random_state=seed, n_init=10).fit_predict(profiles)
    market_cluster_by_id = {
        market_id: int(market_cluster_labels[index]) for index, market_id in enumerate(market_ids)
    }
    oov_cluster = market_cluster_by_id[OOV_MARKET_ID]
    calibration_market_ids = frame.loc[calibration_anchor_mask, "market_id"].astype(str).to_numpy()
    calibration_cluster = np.asarray(
        [market_cluster_by_id.get(market_id, oov_cluster) for market_id in calibration_market_ids], dtype=int
    )
    global_q = _conformal_quantile(calibration_error, CONFORMAL_ALPHA)
    quantiles_by_cluster = {}
    calibration_counts = {}
    for cluster_id in range(n_market_clusters):
        subset = calibration_error[calibration_cluster == cluster_id]
        calibration_counts[str(cluster_id)] = int(len(subset))
        quantiles_by_cluster[str(cluster_id)] = (
            _conformal_quantile(subset, CONFORMAL_ALPHA) if len(subset) >= 10 else global_q
        )

    test_rows = frame.loc[test_mask].copy().reset_index(names="source_row_index")
    test_indices = test_rows["source_row_index"].to_numpy(dtype=int)
    prediction = point[test_indices]
    lower = np.empty(len(test_rows), dtype=float)
    upper = np.empty(len(test_rows), dtype=float)
    for index, (_, row) in enumerate(test_rows.iterrows()):
        cluster_id = str(market_cluster_by_id.get(str(row["market_id"]), oov_cluster))
        qhat = quantiles_by_cluster.get(cluster_id, global_q)
        source_index = int(row["source_row_index"])
        lower[index] = max(float(lower_prices[source_index]), prediction[index] * np.exp(-qhat))
        upper[index] = min(float(upper_prices[source_index]), prediction[index] * np.exp(qhat))
    actual_test = test_rows["synthetic_latent_true_price_inr"].to_numpy(dtype=float)
    interval_coverage = (actual_test >= lower) & (actual_test <= upper)
    interval_width = upper - lower
    prior_price = np.exp(mu_array[test_indices])
    output = pd.DataFrame(
        {
            "synthetic_property_id": test_rows["synthetic_property_id"],
            "sro_area": test_rows["sro_area"],
            "micro_market_id": test_rows["micro_market_id"],
            "micro_market": test_rows["micro_market"],
            "synthetic_quarter": test_rows["synthetic_quarter"],
            "synthetic_asking_price_inr": test_rows["synthetic_asking_price_inr"].round().astype("int64"),
            "synthetic_declared_value_inr": test_rows["synthetic_declared_registration_value_inr"].round().astype("int64"),
            "synthetic_guidance_value_inr": test_rows["synthetic_area_derived_guideline_value_inr"].round().astype("int64"),
            "synthetic_latent_true_price_inr_for_simulation_scoring": actual_test.round().astype("int64"),
            "blp_graph_estimated_latent_price_inr": prediction.round().astype("int64"),
            "prior_log_price_hedonic_component": hedonic_array[test_indices],
            "prior_log_price_locality_premium": locality_array[test_indices],
            "estimated_asking_markup_fraction": asking_markup_array[test_indices],
            "estimated_underdeclaration_probability": underdeclaration_probability_array[test_indices],
            "functional_market_cluster": [market_cluster_by_id.get(str(value), oov_cluster) for value in test_rows["market_id"]],
            "lognormal_prior_price_inr": prior_price.round().astype("int64"),
            "conformal_90_lower_inr": lower.round().astype("int64"),
            "conformal_90_upper_inr": upper.round().astype("int64"),
            "synthetic_validation_sample_flag": test_rows["validation_sample_flag"].astype(int),
        }
    )

    predictions = {
        "asking_only": test_rows["synthetic_asking_price_inr"].to_numpy(dtype=float),
        "declared_only": test_rows["synthetic_declared_registration_value_inr"].to_numpy(dtype=float),
        "naive_geometric_pool": np.sqrt(
            test_rows["synthetic_asking_price_inr"].to_numpy(dtype=float)
            * test_rows["synthetic_declared_registration_value_inr"].to_numpy(dtype=float)
        ),
        "bounded_latent_graph": prediction,
    }
    model_metrics = {name: _metrics(actual_test, pred) for name, pred in predictions.items()}
    interval_metrics = {
        "nominal_coverage_percent": 100 * (1 - CONFORMAL_ALPHA),
        "empirical_coverage_percent": round(float(interval_coverage.mean() * 100), 3),
        "mean_width_inr": round(float(interval_width.mean()), 2),
        "mean_width_percent_of_true_price": round(float((interval_width / actual_test).mean() * 100), 3),
        "calibration_anchor_count": int(calibration_anchor_mask.sum()),
        "calibration_anchor_count_by_functional_cluster": calibration_counts,
        "conformal_quantile_log_scale_by_functional_cluster": {key: round(float(value), 6) for key, value in quantiles_by_cluster.items()},
    }
    by_area = {}
    for sro, indices in test_rows.groupby("sro_area", sort=True).indices.items():
        indices = np.asarray(indices, dtype=int)
        by_area[str(sro)] = {
            "rows": int(len(indices)),
            "blp_graph_metrics": _metrics(actual_test[indices], prediction[indices]),
            "asking_only_metrics": _metrics(actual_test[indices], predictions["asking_only"][indices]),
            "declared_only_metrics": _metrics(actual_test[indices], predictions["declared_only"][indices]),
            "conformal_90_coverage_percent": round(float(interval_coverage[indices].mean() * 100), 3),
            "conformal_90_mean_width_inr": round(float(interval_width[indices].mean()), 2),
        }

    indices_rows = []
    for (market_id, market_name, sro, quarter), group in test_rows.groupby(
        ["micro_market_id", "micro_market", "sro_area", "synthetic_quarter"], sort=True
    ):
        positions = test_rows.index.get_indexer(group.index)
        actual_group = actual_test[positions]
        estimate_group = prediction[positions]
        ask_group = group["synthetic_asking_price_inr"].to_numpy(dtype=float)
        deed_group = group["synthetic_declared_registration_value_inr"].to_numpy(dtype=float)
        indices_rows.append(
            {
                "micro_market_id": market_id,
                "micro_market": market_name,
                "sro_area": sro,
                "quarter": quarter,
                "rows": int(len(group)),
                "estimated_asking_markup_fraction": float(np.mean(asking_markup_array[test_indices][positions])),
                "simulated_true_asking_markup_fraction_for_error_check": float(np.mean(ask_group / actual_group - 1.0)),
                "estimated_underdeclaration_probability": float(np.mean(underdeclaration_probability_array[test_indices][positions])),
                "simulated_true_underdeclaration_probability_for_error_check": float(np.mean(deed_group < actual_group)),
                "reliability": "sparse" if len(group) < 5 else "group_estimate",
            }
        )

    report = {
        "status": "simulation_only_architecture_prototype",
        "data_status": "ALL model training labels, paired signals, and validation values in this run are generated synthetic fixtures; not observed Bengaluru transactions.",
        "evaluation_quarter": TEST_QUARTER,
        "random_seed": int(seed),
        "functional_similarity_graph_enabled": bool(use_graph),
        "prior_quarter_gru_enabled": bool(use_temporal),
        "training_quarters": TRAIN_QUARTERS,
        "conformal_calibration_quarter": CALIBRATION_QUARTER,
        "training_rows": int(train_mask.sum()),
        "synthetic_validation_anchors_in_training": int(anchor_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "synthetic_validation_anchors_for_conformal": int(calibration_anchor_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "micro_market_nodes": int(len(market_ids) - 1),
        "unknown_market_policy": "markets absent from training are mapped to a shared OOV node; graph profiles are fit on training-period rows only",
        "functional_similarity_graph_undirected_edges": int(len(edges)),
        "graph_edge_type": "top-5 covariate-profile similarity; not spatial adjacency",
        "temporal_module": "GRU over prior-quarter synthetic asking/guidance and declared/guidance summaries; current and future quarters are excluded from each row's history",
        "model_description": "Native PyTorch additive spline GAM hedonic head plus multi-head attention over training-profile-only functional micro-market similarity edges and a GRU over prior-quarter asking/guidance and declared/guidance summaries. Unseen markets map to a shared OOV node. Training combines interval-censored likelihood with generated exact-price anchors; inference uses a bounded truncated-lognormal posterior mean.",
        "learned_log_price_scale_sigma": round(sigma, 6),
        "censored_objective_trace": loss_history,
        "holdout_metrics_against_synthetic_latent_truth": model_metrics,
        "conformal_90_metrics_against_synthetic_latent_truth": interval_metrics,
        "holdout_metrics_by_sro": by_area,
        "synthetic_market_quarter_indices": indices_rows,
        "_calibration_bundle": {
            "predicted_prices": calibration_estimated.astype(float).tolist(),
            "actual_prices": calibration_actual.astype(float).tolist(),
            "cluster_ids": calibration_cluster.astype(int).tolist(),
        },
        "interpretation": {
            "hedonic_component": "The network emits a unit-attribute log-price component.",
            "locality_premium": "The network emits a graph/GRU log-price component; their sum is the latent log-price prior.",
            "asking_discount": "A nonnegative asking markup fraction is learned from flagged synthetic anchors using asking/reference - 1 and unit, graph, and temporal context; each market-quarter estimate is aggregated from row predictions.",
            "underdeclaration_probability": "A Bernoulli probability is learned from flagged synthetic anchors where the declared value is below generated reference price; each market-quarter estimate is aggregated from row predictions.",
        },
        "limitations": [
            "Synthetic fixture values are generated and cannot establish real-market accuracy or policy indices.",
            "The graph has unit and training-known micro-market representations plus functional covariate-similarity edges; unseen markets map to an OOV node. No coordinates, POI nodes, RERA project nodes, metro-event edges, or true spatial-neighbor edges are available.",
            "Functional profiles and profile scaling use training-period covariates only; markets absent from training use a shared OOV node.",
            "Conformal strata are six functional market clusters built from synthetic property profiles because flagged anchors are too sparse per market; these are not geographic clusters.",
            "Synthetic latent truth is used only on the final quarter to score the simulation and is never used to fit the model outside flagged synthetic anchors.",
            "MugRep, ReGram, ST-RAP, LightGBM, and SHAP benchmarks are not run in this prototype; their packages or licensed data/implementations are not present.",
        ],
    }

    if return_predictor:
        report["_predictor"] = {
            "model": model,
            "feature_encoder": unit_encoder,
            "profiles": profile_tensor,
            "adjacency": adjacency_tensor,
            "history": history_tensor,
            "market_ids": market_ids,
            "market_to_index": market_to_index,
            "oov_market_id": OOV_MARKET_ID,
            "market_cluster_by_id": market_cluster_by_id,
            "quantiles_by_cluster": quantiles_by_cluster,
            "global_quantile": global_q,
            "feature_groups": unit_encoder.feature_groups,
            "sigma": sigma,
        }

    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    if write_outputs:
        processed.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)
        output.to_csv(processed / "synthetic_blp_graph_holdout_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(edges).to_csv(processed / "synthetic_functional_market_graph.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(indices_rows).to_csv(processed / "synthetic_market_quarter_indices.csv", index=False, encoding="utf-8-sig")
        serializable = {key: value for key, value in report.items() if not key.startswith("_")}
        (reports / "synthetic_blp_graph_metrics.json").write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (reports / "synthetic_blp_graph_metrics.md").write_text(_format_report(report), encoding="utf-8")
    return report


def _format_report(report: dict) -> str:
    lines = [
        "# BLP-Graph simulation prototype",
        "",
        f"**Status:** `{report['status']}`",
        "",
        f"> {report['data_status']}",
        "",
        "## Simulation design",
        "",
        f"- Training: {', '.join(report['training_quarters'])} ({report['training_rows']:,} generated rows; {report['synthetic_validation_anchors_in_training']:,} generated exact-price anchors).",
        f"- Conformal calibration: {report['conformal_calibration_quarter']} ({report['synthetic_validation_anchors_for_conformal']:,} generated validation anchors).",
        f"- Holdout: {report['evaluation_quarter']} ({report['test_rows']:,} generated rows).",
        f"- Graph: {report['micro_market_nodes']:,} micro-market nodes and {report['functional_similarity_graph_undirected_edges']:,} undirected functional similarity edges.",
        f"- Model: {report['model_description']}",
        "",
        "## Synthetic holdout results",
        "",
        "Metrics below compare model outputs with generated latent prices in the simulation fixture. They are not empirical Bengaluru price-prediction results.",
        "",
        "| Signal/model | MAE (INR lakh) | MAPE | Median APE | Hit within 10% | Hit within 20% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "asking_only": "Asking price only",
        "declared_only": "Declared value only",
        "naive_geometric_pool": "Naive geometric pool",
        "bounded_latent_graph": "Bounded latent graph",
    }
    for key, label in labels.items():
        metric = report["holdout_metrics_against_synthetic_latent_truth"][key]
        lines.append(
            f"| {label} | {metric['mae_inr'] / 100000:,.2f} | {metric['mape_percent']:.2f}% | {metric['median_ape_percent']:.2f}% | {metric['hit_rate_within_10_percent']:.1f}% | {metric['hit_rate_within_20_percent']:.1f}% |"
        )
    interval = report["conformal_90_metrics_against_synthetic_latent_truth"]
    ablations = report.get("ablation_metrics", {})
    seed_summary = report.get("five_seed_summary", {}).get("mape_percent", {})
    wilcoxon_result = report.get("paired_wilcoxon_vs_geometric_pool", {})
    same_quarter = report.get("same_quarter_conformal_metrics_against_synthetic_latent_truth")
    full_seed_label = "five-seed mean prediction" if report.get("five_seed_summary") else "single-seed run"
    no_graph_mape = ablations.get("no_functional_graph_edges", {}).get("mape_percent")
    no_gru_mape = ablations.get("no_gru_temporal_module", {}).get("mape_percent")
    p_value = wilcoxon_result.get("p_value")
    lines.extend(
        [
            "",
            "## Baselines, ablations, and stability",
            "",
            "| Variant | MAPE |",
            "|---|---:|",
            f"| Full graph + GRU ({full_seed_label}) | {report['holdout_metrics_against_synthetic_latent_truth']['bounded_latent_graph']['mape_percent']:.2f}% |",
            f"| No functional graph edges | {f'{no_graph_mape:.2f}%' if no_graph_mape is not None else 'Pending'} |",
            f"| No GRU temporal module | {f'{no_gru_mape:.2f}%' if no_gru_mape is not None else 'Pending'} |",
            *[
                f"| {key.replace('_', ' ')} | {metric['mape_percent']:.2f}% |"
                for key, metric in ablations.items()
                if key not in ("no_functional_graph_edges", "no_gru_temporal_module")
            ],
            *[
                f"| {key.replace('_', ' ')} | {metric['mape_percent']:.2f}% |"
                for key, metric in report.get("anchor_only_baseline_metrics", {}).items()
            ],
            "",
            f"Five-seed MAPE standard deviation: {seed_summary.get('standard_deviation', float('nan')):.3f} percentage points. Paired market-level Wilcoxon comparison with the geometric-pool baseline: p={p_value if p_value is not None else float('nan'):.4g} across {wilcoxon_result.get('n', 0):,} synthetic micro-markets.",
            (
                f"Leave-micro-market-out simulation: {report['leave_micro_market_out_evaluation']['metrics_against_synthetic_latent_truth']['mape_percent']:.2f}% MAPE on {report['leave_micro_market_out_evaluation']['test_rows']:,} held-out rows from {report['leave_micro_market_out_evaluation']['held_out_markets_with_q1_records']:,} held-out markets."
                if report.get("leave_micro_market_out_evaluation") else "Leave-micro-market-out simulation: pending."
            ),
            "",
            "## Synthetic 90% conformal intervals",
            "",
            f"- Prior-quarter calibration ({report['conformal_calibration_quarter']} to {report['evaluation_quarter']}): {interval['empirical_coverage_percent']:.1f}% coverage (nominal {interval['nominal_coverage_percent']:.0f}%).",
            (
                f"- Same-quarter calibration on flagged validation anchors: {same_quarter['empirical_coverage_percent']:.1f}% coverage on {same_quarter['evaluation_rows']:,} remaining synthetic rows."
                if same_quarter
                else "- Same-quarter calibration: pending aggregate simulation run."
            ),
            "- Calibration is stratified by functional-similarity cluster, not geography; the data contains no coordinates.",
            "- Same-quarter calibration illustrates exchangeability; prior-quarter undercoverage shows that future-quarter coverage is not established.",
            "",
            "## Scope and limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "The run implements the synthetic-data versions of the dual-signal censored likelihood, unit-attribute spline GAM, functional micro-market attention, prior-quarter GRU, learned asking markup and under-declaration heads, an additive hedonic/locality decomposition, and cluster-stratified split conformal intervals. It does not complete the proposal's real spatial graph or empirical validation because RERA, POI/metro, geocodes, and observed transaction records are absent.",
            "",
            "Rebuild with `python scripts/run_blp_graph_simulation.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def _anchor_only_baselines(frame: pd.DataFrame) -> dict:
    """Run transparent tabular proxies using only flagged synthetic anchors."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    train = frame["synthetic_quarter"].isin(TRAIN_QUARTERS).to_numpy()
    anchors = train & frame["validation_sample_flag"].eq(1).to_numpy()
    holdout = frame["synthetic_quarter"].eq(TEST_QUARTER).to_numpy()
    x, _, _, _, _, _ = _prepare_features(frame, train)
    truth = frame.loc[holdout, "synthetic_latent_true_price_inr"].to_numpy(dtype=float)
    lower = np.maximum(
        frame.loc[holdout, "synthetic_declared_registration_value_inr"].to_numpy(dtype=float),
        frame.loc[holdout, "synthetic_area_derived_guideline_value_inr"].to_numpy(dtype=float),
    )
    upper = frame.loc[holdout, "synthetic_asking_price_inr"].to_numpy(dtype=float)
    result = {}
    models = {
        "hedonic_gam_ridge_anchor_proxy": Ridge(alpha=10.0),
        "hist_gradient_boosting_anchor_proxy": HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, min_samples_leaf=12,
            l2_regularization=5.0, random_state=SEED,
        ),
    }
    for label, estimator in models.items():
        estimator.fit(x[anchors], np.log(frame.loc[anchors, "synthetic_latent_true_price_inr"].to_numpy(dtype=float)))
        raw = np.exp(estimator.predict(x[holdout]))
        bounded = np.clip(raw, lower, upper)
        result[label] = _metrics(truth, bounded)
        result[label]["training_anchor_count"] = int(anchors.sum())
        result[label]["description"] = (
            "Spline-feature Ridge hedonic proxy; generated anchor labels only."
            if label == "hedonic_gam_ridge_anchor_proxy"
            else "Histogram gradient boosting tabular proxy; not LightGBM, generated anchor labels only."
        )
    return result


def run_synthetic_blp_graph(project_root: Path) -> dict:
    """Run the proposal-aligned simulation workflow on the two supplied generated CSVs."""
    frame = _load_synthetic(project_root)
    seeds = [2026, 2027, 2028, 2029, 2030]
    full_runs = []
    prediction_frames = []
    for seed in seeds:
        run_report = _fit_graph_model(frame, project_root, seed=seed)
        full_runs.append(run_report)
        prediction_frames.append(
            pd.read_csv(project_root / "data" / "processed" / "synthetic_blp_graph_holdout_predictions.csv")
        )

    primary = prediction_frames[0].copy()
    prediction_matrix = np.vstack(
        [run["blp_graph_estimated_latent_price_inr"].to_numpy(dtype=float) for run in prediction_frames]
    )
    mean_prediction = prediction_matrix.mean(axis=0)
    actual = primary["synthetic_latent_true_price_inr_for_simulation_scoring"].to_numpy(dtype=float)
    geometric = np.sqrt(
        primary["synthetic_asking_price_inr"].to_numpy(dtype=float)
        * primary["synthetic_declared_value_inr"].to_numpy(dtype=float)
    )

    ablations = {}
    for label, graph, temporal, censoring, asking, deeds in [
        ("no_functional_graph_edges", False, True, True, True, True),
        ("no_gru_temporal_module", True, False, True, True, True),
        ("no_censored_likelihood", True, True, False, True, True),
        ("no_asking_signal", True, True, True, False, True),
        ("no_deed_signal", True, True, True, True, False),
    ]:
        ablation_report = _fit_graph_model(
            frame,
            project_root,
            seed=SEED,
            use_graph=graph,
            use_temporal=temporal,
            use_censoring=censoring,
            use_asking=asking,
            use_deeds=deeds,
        )
        ablation_frame = pd.read_csv(
            project_root / "data" / "processed" / "synthetic_blp_graph_holdout_predictions.csv"
        )
        ablations[label] = _metrics(
            actual,
            ablation_frame["blp_graph_estimated_latent_price_inr"].to_numpy(dtype=float),
        )

    from scipy.stats import wilcoxon

    market_errors = pd.DataFrame(
        {
            "micro_market_id": primary["micro_market_id"].astype(str).to_numpy(),
            "graph_ape": np.abs(mean_prediction - actual) / actual.clip(min=1),
            "pool_ape": np.abs(geometric - actual) / actual.clip(min=1),
        }
    ).groupby("micro_market_id", sort=True)[["graph_ape", "pool_ape"]].mean()
    try:
        test = wilcoxon(
            market_errors["graph_ape"],
            market_errors["pool_ape"],
            alternative="two-sided",
            zero_method="wilcox",
        )
        wilcoxon_result = {
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "n": int(len(market_errors)),
            "comparison": "paired micro-market mean absolute percentage error: five-seed mean BLP-Graph vs naive geometric pool",
            "unit_of_analysis": "micro-market",
            "status": "simulation_only_market_level_inference",
        }
    except ValueError as exc:
        wilcoxon_result = {
            "statistic": None,
            "p_value": None,
            "n": int(len(market_errors)),
            "comparison": "paired micro-market mean absolute percentage error: five-seed mean BLP-Graph vs naive geometric pool",
            "unit_of_analysis": "micro-market",
            "status": f"not_computed: {exc}",
        }

    aggregate = full_runs[0]
    aggregate["holdout_metrics_against_synthetic_latent_truth"]["bounded_latent_graph"] = _metrics(
        actual, mean_prediction
    )
    seed_mape = [
        run["holdout_metrics_against_synthetic_latent_truth"]["bounded_latent_graph"]["mape_percent"]
        for run in full_runs
    ]
    seed_mae = [
        run["holdout_metrics_against_synthetic_latent_truth"]["bounded_latent_graph"]["mae_inr"]
        for run in full_runs
    ]
    aggregate["five_seed_summary"] = {
        "seeds": seeds,
        "mape_percent": {
            "mean": float(np.mean(seed_mape)),
            "standard_deviation": float(np.std(seed_mape, ddof=1)),
            "per_seed": seed_mape,
        },
        "mae_inr": {
            "mean": float(np.mean(seed_mae)),
            "standard_deviation": float(np.std(seed_mae, ddof=1)),
            "per_seed": seed_mae,
        },
        "prediction_aggregation": "mean of five independently initialized BLP-Graph prediction vectors",
    }
    aggregate["ablation_metrics"] = ablations
    aggregate["anchor_only_baseline_metrics"] = _anchor_only_baselines(frame)
    aggregate["paired_wilcoxon_vs_geometric_pool"] = wilcoxon_result
    aggregate["learned_log_price_scale_sigma"] = float(
        np.mean([run["learned_log_price_scale_sigma"] for run in full_runs])
    )

    all_markets = sorted(frame["market_id"].astype(str).unique().tolist())
    market_rng = np.random.default_rng(SEED)
    held_out_count = max(1, int(np.ceil(0.20 * len(all_markets))))
    leave_out_markets = set(market_rng.choice(all_markets, size=held_out_count, replace=False).tolist())
    leave_out_report = _fit_graph_model(
        frame,
        project_root,
        seed=SEED,
        excluded_markets=leave_out_markets,
    )
    aggregate["leave_micro_market_out_evaluation"] = {
        "held_out_market_count": int(len(leave_out_markets)),
        "held_out_markets_with_q1_records": int(
            frame.loc[frame["synthetic_quarter"].eq(TEST_QUARTER) & frame["market_id"].isin(leave_out_markets), "market_id"].nunique()
        ),
        "test_rows": int(leave_out_report["test_rows"]),
        "metrics_against_synthetic_latent_truth": leave_out_report["holdout_metrics_against_synthetic_latent_truth"]["bounded_latent_graph"],
        "functional_similarity_graph_enabled": True,
        "interpretation": "Held-out markets are absent from fitted profiles and graph nodes; their rows map to the shared OOV node. Only their per-property input features are used at prediction time.",
    }

    calibration_bundles = [run["_calibration_bundle"] for run in full_runs]
    calibration_actual = np.asarray(calibration_bundles[0]["actual_prices"], dtype=float)
    calibration_cluster = np.asarray(calibration_bundles[0]["cluster_ids"], dtype=int)
    calibration_prediction_matrix = np.vstack(
        [np.asarray(bundle["predicted_prices"], dtype=float) for bundle in calibration_bundles]
    )
    calibration_mean_prediction = calibration_prediction_matrix.mean(axis=0)
    calibration_error = np.abs(
        np.log(calibration_actual.clip(min=1)) - np.log(calibration_mean_prediction.clip(min=1))
    )
    ensemble_global_q = _conformal_quantile(calibration_error, CONFORMAL_ALPHA)
    seed_cluster_keys = full_runs[0]["conformal_90_metrics_against_synthetic_latent_truth"][
        "calibration_anchor_count_by_functional_cluster"
    ]
    ensemble_quantiles = {}
    ensemble_calibration_counts = {}
    for cluster_key in seed_cluster_keys:
        cluster_id = int(cluster_key)
        subset = calibration_error[calibration_cluster == cluster_id]
        ensemble_calibration_counts[str(cluster_id)] = int(len(subset))
        ensemble_quantiles[str(cluster_id)] = (
            _conformal_quantile(subset, CONFORMAL_ALPHA) if len(subset) >= 10 else ensemble_global_q
        )

    primary_estimate = primary["blp_graph_estimated_latent_price_inr"].to_numpy(dtype=float)
    lower_bound = np.maximum(
        primary["synthetic_declared_value_inr"].to_numpy(dtype=float),
        primary["synthetic_guidance_value_inr"].to_numpy(dtype=float),
    )
    upper_bound = primary["synthetic_asking_price_inr"].to_numpy(dtype=float)
    test_clusters = primary["functional_market_cluster"].astype(str).to_numpy()
    test_quantiles = np.asarray(
        [ensemble_quantiles.get(cluster, ensemble_global_q) for cluster in test_clusters], dtype=float
    )
    lower = np.maximum(lower_bound, mean_prediction * np.exp(-test_quantiles))
    upper = np.minimum(upper_bound, mean_prediction * np.exp(test_quantiles))
    primary["blp_graph_primary_seed_estimate_inr"] = primary_estimate.round().astype("int64")
    primary["blp_graph_estimated_latent_price_inr"] = mean_prediction.round().astype("int64")
    primary["conformal_90_lower_inr"] = lower.round().astype("int64")
    primary["conformal_90_upper_inr"] = upper.round().astype("int64")
    coverage = (actual >= lower) & (actual <= upper)
    width = upper - lower
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["empirical_coverage_percent"] = float(
        coverage.mean() * 100
    )
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["mean_width_inr"] = float(width.mean())
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["mean_width_percent_of_true_price"] = float(
        (width / actual.clip(min=1)).mean() * 100
    )
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["calibration_anchor_count"] = int(len(calibration_error))
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["calibration_anchor_count_by_functional_cluster"] = ensemble_calibration_counts
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["conformal_quantile_log_scale_by_functional_cluster"] = {
        key: round(float(value), 6) for key, value in ensemble_quantiles.items()
    }
    aggregate["conformal_90_metrics_against_synthetic_latent_truth"]["calibration_model"] = "five-seed mean, recalibrated on 2023Q4 anchors"

    same_quarter_calibration_mask = primary["synthetic_validation_sample_flag"].to_numpy(dtype=int) == 1
    same_quarter_evaluation_mask = ~same_quarter_calibration_mask
    same_quarter_calibration_error = np.abs(
        np.log(actual[same_quarter_calibration_mask].clip(min=1))
        - np.log(mean_prediction[same_quarter_calibration_mask].clip(min=1))
    )
    same_quarter_calibration_sro = primary.loc[same_quarter_calibration_mask, "sro_area"].astype(str).to_numpy()
    same_quarter_global_q = _conformal_quantile(same_quarter_calibration_error, CONFORMAL_ALPHA)
    same_quarter_quantiles = {}
    same_quarter_calibration_counts = {}
    for sro in sorted(primary["sro_area"].astype(str).unique()):
        group_errors = same_quarter_calibration_error[same_quarter_calibration_sro == sro]
        same_quarter_calibration_counts[sro] = int(len(group_errors))
        same_quarter_quantiles[sro] = (
            _conformal_quantile(group_errors, CONFORMAL_ALPHA)
            if len(group_errors) >= 10
            else same_quarter_global_q
        )
    in_quarter_lower = np.empty(len(primary), dtype=float)
    in_quarter_upper = np.empty(len(primary), dtype=float)
    for position, row in primary.iterrows():
        qhat = same_quarter_quantiles.get(str(row["sro_area"]), same_quarter_global_q)
        point = mean_prediction[position]
        in_quarter_lower[position] = max(float(row["synthetic_declared_value_inr"]), point * np.exp(-qhat))
        in_quarter_upper[position] = min(float(row["synthetic_asking_price_inr"]), point * np.exp(qhat))
    same_q_coverage = (
        (actual[same_quarter_evaluation_mask] >= in_quarter_lower[same_quarter_evaluation_mask])
        & (actual[same_quarter_evaluation_mask] <= in_quarter_upper[same_quarter_evaluation_mask])
    )
    same_q_width = in_quarter_upper[same_quarter_evaluation_mask] - in_quarter_lower[same_quarter_evaluation_mask]
    aggregate["same_quarter_conformal_metrics_against_synthetic_latent_truth"] = {
        "nominal_coverage_percent": 100 * (1 - CONFORMAL_ALPHA),
        "empirical_coverage_percent": float(same_q_coverage.mean() * 100),
        "mean_width_inr": float(same_q_width.mean()),
        "mean_width_percent_of_true_price": float(
            (same_q_width / actual[same_quarter_evaluation_mask].clip(min=1)).mean() * 100
        ),
        "calibration_anchor_count": int(same_quarter_calibration_mask.sum()),
        "evaluation_rows": int(same_quarter_evaluation_mask.sum()),
        "calibration_anchor_count_by_sro": same_quarter_calibration_counts,
        "calibration_quantile_log_scale_by_sro": {
            key: float(value) for key, value in same_quarter_quantiles.items()
        },
        "design": "calibrate on flagged synthetic validation anchors from 2024Q1; score the remaining synthetic rows in 2024Q1",
        "interpretation": "same-quarter exchangeability demonstration only; does not validate next-quarter coverage",
    }
    primary["same_quarter_conformal_90_lower_inr"] = in_quarter_lower.round().astype("int64")
    primary["same_quarter_conformal_90_upper_inr"] = in_quarter_upper.round().astype("int64")

    indices = []
    for (market_id, market_name, sro, quarter), positions in primary.groupby(
        ["micro_market_id", "micro_market", "sro_area", "synthetic_quarter"], sort=True
    ).indices.items():
        positions = np.asarray(positions, dtype=int)
        ask_values = primary.iloc[positions]["synthetic_asking_price_inr"].to_numpy(dtype=float)
        deed_values = primary.iloc[positions]["synthetic_declared_value_inr"].to_numpy(dtype=float)
        indices.append(
            {
                "micro_market_id": str(market_id),
                "micro_market": str(market_name),
                "sro_area": str(sro),
                "quarter": str(quarter),
                "rows": int(len(positions)),
                "estimated_asking_markup_fraction": float(primary.iloc[positions]["estimated_asking_markup_fraction"].mean()),
                "simulated_true_asking_markup_fraction_for_error_check": float(np.mean(ask_values / actual[positions] - 1.0)),
                "estimated_underdeclaration_probability": float(primary.iloc[positions]["estimated_underdeclaration_probability"].mean()),
                "simulated_true_underdeclaration_probability_for_error_check": float(
                    np.mean(deed_values < actual[positions])
                ),
                "reliability": "sparse" if len(positions) < 5 else "group_estimate",
            }
        )
    aggregate["synthetic_market_quarter_indices"] = indices
    aggregate["evaluation_protocols"] = {
        "chronological_holdout": "Train 2023Q2-Q3; calibrate on flagged generated anchors in 2023Q4; evaluate on 2024Q1.",
        "leave_micro_market_out": "Randomly select 20% of micro-market IDs; fit encoders, profiles, and graph only on the remaining training markets; map excluded markets to a shared OOV node; score their 2024Q1 records.",
        "random_seeds": seeds,
        "conformal_alpha": CONFORMAL_ALPHA,
        "conformal_strata": "six functional market clusters (or fewer if the fixture has fewer market IDs), using generated calibration anchors",
    }

    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    primary.to_csv(processed / "synthetic_blp_graph_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(indices).to_csv(processed / "synthetic_market_quarter_indices.csv", index=False, encoding="utf-8-sig")
    (reports / "synthetic_blp_graph_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (reports / "synthetic_blp_graph_metrics.md").write_text(
        _format_report(aggregate), encoding="utf-8"
    )
    return aggregate
