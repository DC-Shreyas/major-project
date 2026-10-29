"""Local HTTP server for the synthetic property valuation explorer."""

from __future__ import annotations

import argparse
import json
import mimetypes
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
import torch

from .synthetic_blp_graph import (
    CONFORMAL_ALPHA,
    TEST_QUARTER,
    _fit_graph_model,
    _load_synthetic,
    _posterior_price,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
QUARTER_ORDER = ("2023Q2", "2023Q3", "2023Q4", "2024Q1")
EVALUATION_QUARTER = TEST_QUARTER


def _as_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


class PropertyExplorer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.records = _load_synthetic(project_root)
        fitted = _fit_graph_model(
            self.records,
            project_root,
            seed=2026,
            return_predictor=True,
            write_outputs=False,
        )
        self.predictor = fitted.pop("_predictor")
        # Report the score for the same single-seed model instance served below.
        # Reading the project-level ensemble report here made the UI score stale
        # or inconsistent with the estimator actually returned by this server.
        aggregate = fitted
        graph_metric = aggregate["holdout_metrics_against_synthetic_latent_truth"]["bounded_latent_graph"]
        interval = aggregate["conformal_90_metrics_against_synthetic_latent_truth"]
        self.report = {
            "status": aggregate["status"],
            "data_source": "dataset/synthetic_property_valuation_data.csv + dataset/synthetic_registered_deeds_validation.csv",
            "records": int(len(self.records)),
            "distinct_micro_markets": int(self.records["micro_market_id"].nunique()),
            "sro_areas": sorted(self.records["sro_area"].astype(str).unique().tolist()),
            "property_types": sorted(self.records["property_type"].astype(str).unique().tolist()),
            "quarters": list(QUARTER_ORDER),
            "estimate_quarters": [EVALUATION_QUARTER],
            "training_quarters": list(aggregate["training_quarters"]),
            "training_rows": int(aggregate["training_rows"]),
            "served_random_seed": int(aggregate["random_seed"]),
            "calibration_quarter": aggregate["conformal_calibration_quarter"],
            "calibration_rows": int(aggregate["synthetic_validation_anchors_for_conformal"]),
            "evaluation_quarter": aggregate["evaluation_quarter"],
            "evaluation_rows": int(aggregate["test_rows"]),
            "model": aggregate["model_description"],
            "holdout_metrics_against_synthetic_latent_truth": graph_metric,
            "calibration": {
                "nominal_coverage_percent": interval["nominal_coverage_percent"],
                "observed_coverage_percent": interval["empirical_coverage_percent"],
                "mean_width_inr": interval["mean_width_inr"],
                "reference_rows": int(aggregate["synthetic_validation_anchors_for_conformal"]),
                "reference_quarter": aggregate["conformal_calibration_quarter"],
            },
            "scope_note": aggregate["data_status"] + " Estimates are demonstrations, not appraisals.",
        }
        self.model = self
        self._options = self._build_market_options()

    def _build_market_options(self) -> list[dict]:
        source = self.records.loc[
            self.records["synthetic_quarter"].isin(("2023Q2", "2023Q3"))
        ]
        fields = [
            "sro_area", "micro_market_id", "micro_market", "property_type",
        ]
        grouped = source.groupby(fields, dropna=False, sort=True)
        result = []
        for key, group in grouped:
            result.append({
                "sro_area": str(key[0]),
                "micro_market_id": str(key[1]),
                "micro_market": str(key[2]),
                "property_type": str(key[3]),
                "guideline_rate_inr_per_sqm": _as_int(group["guideline_rate_inr_per_sqm"].median()),
                "typical_area_sqft": _as_int(group["synthetic_area_sqft"].median()),
                "median_asking_price_inr": _as_int(group["synthetic_asking_price_inr"].median()),
                "median_declared_price_inr": _as_int(group["synthetic_declared_registration_value_inr"].median()),
                "typical_bedrooms": None if group["bedrooms"].isna().all() else round(float(group["bedrooms"].median()), 1),
                "typical_age_years": None if group["building_age_years"].isna().all() else _as_int(group["building_age_years"].median()),
                "records": int(len(group)),
            })
        return result

    def filtered(self, params: dict[str, list[str]]) -> pd.DataFrame:
        frame = self.records
        sro = (params.get("sro") or [""])[0]
        property_type = (params.get("type") or [""])[0]
        quarter = (params.get("quarter") or [""])[0]
        search = (params.get("q") or [""])[0].strip().casefold()
        if sro:
            frame = frame.loc[frame["sro_area"].eq(sro)]
        if property_type:
            frame = frame.loc[frame["property_type"].eq(property_type)]
        if quarter:
            frame = frame.loc[frame["synthetic_quarter"].eq(quarter)]
        if search:
            mask = (
                frame["micro_market"].astype(str).str.casefold().str.contains(search, regex=False)
                | frame["locality"].astype(str).str.casefold().str.contains(search, regex=False)
                | frame["synthetic_property_id"].astype(str).str.casefold().str.contains(search, regex=False)
            )
            frame = frame.loc[mask]
        return frame

    def summary(self, params: dict[str, list[str]]) -> dict:
        frame = self.filtered(params)
        if frame.empty:
            return {
                "rows": 0,
                "median_asking_price_per_sqft_inr": 0,
                "median_asking_price_inr": 0,
                "median_declared_price_inr": 0,
                "trend": [],
                "markets": [],
                "property_mix": [],
                "backtest": self.report["holdout_metrics_against_synthetic_latent_truth"],
                "scope_note": self.report["scope_note"],
            }
        frame = frame.copy()
        frame["asking_price_per_sqft"] = (
            frame["synthetic_asking_price_inr"] / frame["synthetic_area_sqft"]
        )
        trend = []
        for quarter, group in frame.groupby("synthetic_quarter", sort=False):
            trend.append({
                "quarter": str(quarter),
                "guidance_price_inr": _as_int(group["synthetic_area_derived_guideline_value_inr"].median()),
                "asking_price_inr": _as_int(group["synthetic_asking_price_inr"].median()),
                "declared_price_inr": _as_int(group["synthetic_declared_registration_value_inr"].median()),
                "rows": int(len(group)),
            })
        trend.sort(key=lambda item: QUARTER_ORDER.index(item["quarter"]))
        market_rollup = frame.groupby(
            ["sro_area", "micro_market_id", "micro_market"], as_index=False
        ).agg(
            records=("synthetic_property_id", "count"),
            median_asking_price_inr=("synthetic_asking_price_inr", "median"),
            median_asking_price_per_sqft_inr=("asking_price_per_sqft", "median"),
        ).sort_values("median_asking_price_inr", ascending=False).head(8)
        markets = [
            {
                "sro_area": str(row.sro_area),
                "micro_market": str(row.micro_market),
                "records": int(row.records),
                "median_asking_price_inr": _as_int(row.median_asking_price_inr),
                "median_asking_price_per_sqft_inr": round(float(row.median_asking_price_per_sqft_inr), 2),
            }
            for row in market_rollup.itertuples(index=False)
        ]
        mix_rollup = frame.groupby("property_type").size().sort_values(ascending=False)
        property_mix = [
            {"property_type": str(kind), "records": int(count)}
            for kind, count in mix_rollup.items()
        ]
        return {
            "rows": int(len(frame)),
            "median_asking_price_per_sqft_inr": round(float(frame["asking_price_per_sqft"].median()), 2),
            "median_asking_price_inr": _as_int(frame["synthetic_asking_price_inr"].median()),
            "median_declared_price_inr": _as_int(frame["synthetic_declared_registration_value_inr"].median()),
            "trend": trend,
            "markets": markets,
            "property_mix": property_mix,
            "backtest": self.report["holdout_metrics_against_synthetic_latent_truth"],
            "scope_note": self.report["scope_note"],
        }

    def estimate(self, values: dict) -> dict:
        def optional_number(key: str) -> float:
            raw = values.get(key)
            if raw in (None, ""):
                return np.nan
            parsed = float(raw)
            if not np.isfinite(parsed):
                raise ValueError(f"{key} must be finite when supplied.")
            return parsed

        try:
            area = float(values["area_sqft"])
            rate = float(values["guideline_rate_inr_per_sqm"])
            asking = float(values["asking_price_inr"])
            declared = float(values["declared_price_inr"])
            bedrooms = optional_number("bedrooms")
            age = optional_number("building_age_years")
            floor = optional_number("floor_number")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Enter valid numbers for the area, guidance rate, asking price, and declared price.") from exc
        if not np.isfinite([area, rate, asking, declared]).all() or min(area, rate, asking, declared) <= 0:
            raise ValueError("Area, guidance rate, asking price, and declared price must be finite positive numbers.")
        if (np.isfinite(bedrooms) and bedrooms < 0) or (np.isfinite(age) and age < 0) or (np.isfinite(floor) and floor < 0):
            raise ValueError("Bedrooms, age, and floor cannot be negative.")

        sro = str(values.get("sro_area") or "")
        property_type = str(values.get("property_type") or "")
        quarter = str(values.get("as_of_quarter") or EVALUATION_QUARTER)
        if sro not in self.report["sro_areas"] or property_type not in self.report["property_types"]:
            raise ValueError("Choose a covered SRO area and property type.")
        if quarter not in self.report["estimate_quarters"]:
            raise ValueError("This estimator is calibrated only for 2024Q1; choose that supported evaluation quarter.")

        guidance = rate * area / 10.7639
        effective_lower = max(guidance, declared)
        if effective_lower > asking:
            raise ValueError("The higher of the guidance and declared values cannot exceed the asking price.")
        requested_market_id = str(values.get("micro_market_id") or "")
        market_to_index = self.predictor["market_to_index"]
        market_note = None
        if requested_market_id:
            selected_rows = self.records.loc[
                self.records["micro_market_id"].astype(str).eq(requested_market_id)
            ]
            if not selected_rows.empty and not (
                selected_rows["sro_area"].eq(sro) & selected_rows["property_type"].eq(property_type)
            ).any():
                raise ValueError("The selected micro-market does not match the chosen SRO area and property type.")
        if requested_market_id in market_to_index and requested_market_id != self.predictor["oov_market_id"]:
            market_id = requested_market_id
        else:
            market_id = self.predictor["oov_market_id"]
            market_note = "Unseen market: the model used its shared unknown-market context, trained without this market's records."

        model_row = pd.DataFrame([{
            "synthetic_area_sqft": area,
            "bedrooms": bedrooms,
            "building_age_years": age,
            "floor_number": floor,
            "property_type": property_type,
            "sro_area": sro,
        }])
        encoder = self.predictor["feature_encoder"]
        x_tensor = torch.tensor(encoder.transform(model_row), dtype=torch.float32)
        market_tensor = torch.tensor([market_to_index[market_id]], dtype=torch.long)
        quarter_tensor = torch.tensor([QUARTER_ORDER.index(quarter)], dtype=torch.long)
        with torch.no_grad():
            output = self.predictor["model"](
                x_tensor,
                self.predictor["profiles"],
                self.predictor["adjacency"],
                self.predictor["history"],
                market_tensor,
                quarter_tensor,
            )
        mean_log, _, hedonic, locality, _, markup, underdeclaration = output
        mean_value = float(mean_log.item())
        sigma = float(self.predictor["sigma"])
        estimate = float(_posterior_price(
            np.asarray([mean_value]), sigma, np.asarray([effective_lower]), np.asarray([asking])
        )[0])
        cluster_id = str(self.predictor["market_cluster_by_id"][market_id])
        qhat = float(self.predictor["quantiles_by_cluster"].get(
            cluster_id, self.predictor["global_quantile"]
        ))
        hedonic_weights = self.predictor["model"].hedonic_head.weight.detach().cpu().numpy().reshape(-1)
        encoded_row = x_tensor.numpy().reshape(-1)
        hedonic_drivers = {
            name: round(float(np.dot(encoded_row[indices], hedonic_weights[indices])), 5)
            for name, indices in self.predictor["feature_groups"].items()
            if indices
        }
        lower = max(effective_lower, estimate * np.exp(-qhat))
        upper = min(asking, estimate * np.exp(qhat))
        comparables = self._comparables(
            area=area, bedrooms=bedrooms, age=age, sro=sro,
            market_id=market_id, property_type=property_type, quarter=quarter,
        )
        return {
            "estimated_price_inr": int(round(estimate)),
            "interval_lower_inr": int(round(lower)),
            "interval_upper_inr": int(round(upper)),
            "price_per_sqft_inr": round(estimate / area, 2),
            "guidance_value_inr": int(round(guidance)),
            "declared_price_inr": int(round(declared)),
            "asking_price_inr": int(round(asking)),
            "bound_position_percent": round(float((estimate - effective_lower) / max(asking - effective_lower, 1) * 100), 1),
            "hedonic_component_log": round(float(hedonic.item()), 5),
            "hedonic_driver_contributions_log": hedonic_drivers,
            "locality_premium_percent": round(float((np.exp(locality.item()) - 1) * 100), 2),
            "asking_markup_fraction": round(float(markup.item()), 4),
            "underdeclaration_probability": round(float(underdeclaration.item()), 4),
            "calibration": {
                "nominal_coverage_percent": self.report["calibration"]["nominal_coverage_percent"],
                "observed_coverage_percent": round(float(self.report["calibration"]["observed_coverage_percent"]), 2),
                "interval_reference_rows": self.report["calibration"]["reference_rows"],
                "reference_quarter": self.report["calibration"]["reference_quarter"],
            },
            "comparables": comparables,
            "locality_note": market_note,
            "scope_note": "Estimate and interval are produced by the synthetic BLP-Graph workflow; they are not a real-world appraisal.",
        }

    def _comparables(self, *, area: float, bedrooms: float, age: float, sro: str,
                     market_id: str, property_type: str, quarter: str) -> list[dict]:
        quarter_index = QUARTER_ORDER.index(quarter)
        rows = self.records.loc[
            self.records["synthetic_quarter"].map({value: index for index, value in enumerate(QUARTER_ORDER)}).lt(quarter_index)
            & self.records["synthetic_area_sqft"].gt(0)
        ].copy()
        preferred = rows.loc[rows["sro_area"].eq(sro) & rows["property_type"].eq(property_type)]
        rows = preferred if len(preferred) >= 5 else rows
        area_distance = np.abs(np.log(rows["synthetic_area_sqft"].to_numpy(dtype=float) / max(area, 1)))
        bedroom_distance = np.zeros(len(rows), dtype=float)
        if np.isfinite(bedrooms):
            values = rows["bedrooms"].to_numpy(dtype=float)
            bedroom_distance = np.abs(np.nan_to_num(values, nan=bedrooms) - bedrooms) * 0.18
        age_distance = np.zeros(len(rows), dtype=float)
        if np.isfinite(age):
            values = rows["building_age_years"].to_numpy(dtype=float)
            age_distance = np.abs(np.nan_to_num(values, nan=age) - age) / 100
        market_penalty = np.where(rows["micro_market_id"].to_numpy() == market_id, 0.0, 0.12)
        type_penalty = np.where(rows["property_type"].to_numpy() == property_type, 0.0, 0.4)
        rows = rows.assign(_distance=area_distance + bedroom_distance + age_distance + market_penalty + type_penalty)
        rows = rows.sort_values("_distance").head(5)
        return [{
            "micro_market": str(row["micro_market"]),
            "property_type": str(row["property_type"]),
            "quarter": str(row["synthetic_quarter"]),
            "area_sqft": int(row["synthetic_area_sqft"]),
            "asking_price_inr": int(row["synthetic_asking_price_inr"]),
            "declared_price_inr": int(row["synthetic_declared_registration_value_inr"]),
            "similarity_percent": max(0, min(100, round(np.exp(-float(row["_distance"])) * 100))),
        } for _, row in rows.iterrows()]

    def records_page(self, params: dict[str, list[str]]) -> dict:
        frame = self.filtered(params).copy()
        try:
            page = max(1, int((params.get("page") or ["1"])[0]))
        except ValueError:
            page = 1
        try:
            page_size = min(50, max(5, int((params.get("page_size") or ["10"])[0])))
        except ValueError:
            page_size = 10
        frame = frame.sort_values(["synthetic_quarter", "synthetic_property_id"], ascending=[False, True])
        total = len(frame)
        pages = max(1, int(np.ceil(total / page_size)))
        page = min(page, pages)
        start = (page - 1) * page_size
        selected = frame.iloc[start:start + page_size]
        rows = []
        for _, row in selected.iterrows():
            rows.append({
                "id": str(row["synthetic_property_id"]),
                "sro_area": str(row["sro_area"]),
                "micro_market": str(row["micro_market"]),
                "property_type": str(row["property_type"]),
                "quarter": str(row["synthetic_quarter"]),
                "area_sqft": _as_int(row["synthetic_area_sqft"]),
                "asking_price_inr": _as_int(row["synthetic_asking_price_inr"]),
                "declared_price_inr": _as_int(row["synthetic_declared_registration_value_inr"]),
                "guidance_value_inr": _as_int(row["synthetic_area_derived_guidance_value_inr"]),
            })
        return {"rows": rows, "total": int(total), "page": page, "pages": pages, "page_size": page_size}

    def options(self) -> dict:
        return {
            "sro_areas": self.report["sro_areas"],
            "property_types": self.report["property_types"],
            "quarters": self.report["quarters"],
            "estimate_quarters": self.report["estimate_quarters"],
            "default_quarter": EVALUATION_QUARTER,
            "markets": self._options,
            "report": self.report,
        }


class Handler(BaseHTTPRequestHandler):
    explorer: PropertyExplorer

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/api/summary":
            return self._json(200, self.explorer.summary(params))
        if parsed.path == "/api/options":
            return self._json(200, self.explorer.options())
        if parsed.path == "/api/records":
            return self._json(200, self.explorer.records_page(params))
        if parsed.path == "/api/health":
            return self._json(200, {"status": "ok", "records": self.explorer.report["records"]})
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/estimate":
            return self._json(404, {"error": "Route not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("The request body is empty or too large.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.explorer.model.estimate(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        return self._json(200, result)

    def _serve_static(self, request_path: str) -> None:
        relative = unquote(request_path.lstrip("/")) or "index.html"
        target = (WEB_ROOT / relative).resolve()
        if not target.is_relative_to(WEB_ROOT.resolve()) or not target.is_file():
            return self._send(404, b"Not found", "text/plain; charset=utf-8")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def serve(project_root: Path = PROJECT_ROOT, host: str = "127.0.0.1", port: int = 8765) -> None:
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"Website files were not found at {WEB_ROOT}")
    explorer = PropertyExplorer(project_root)
    handler = type("BoundHandler", (Handler,), {"explorer": explorer})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Property explorer is ready at http://{host}:{port}")
    print(f"Loaded {explorer.report['records']:,} paired synthetic records")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping property explorer.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local interactive property explorer.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local computer only)")
    parser.add_argument("--port", default=8765, type=int, help="Port number")
    args = parser.parse_args()
    serve(PROJECT_ROOT, args.host, args.port)


if __name__ == "__main__":
    main()
