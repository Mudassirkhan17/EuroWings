"""
Shared training / inference utilities for competitor price ensemble.
Mirrors logic from final.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_FEATURES = [
    "WeekNumber",
    "WeekSin",
    "WeekCos",
    "Route",
    "CountryRoute",
    "OriginAirport",
    "OriginCountry",
    "DestinationAirport",
    "DestinationCountry",
    "MainAirlineCarrier",
    "IsConnectingFlight",
    "UserCountry",
    "TripType",
    "CabinClass",
    "NumberOfNights",
    "BookingHorizon",
]

CATEGORICAL_FEATURES = [
    "Route",
    "CountryRoute",
    "OriginAirport",
    "OriginCountry",
    "DestinationAirport",
    "DestinationCountry",
    "MainAirlineCarrier",
    "UserCountry",
    "TripType",
    "CabinClass",
]

NUMERIC_FEATURES = [
    "WeekNumber",
    "WeekSin",
    "WeekCos",
    "IsConnectingFlight",
    "NumberOfNights",
    "BookingHorizon",
]

RESIDUAL_FEATURES = [
    col for col in BASE_FEATURES if col not in ["WeekNumber", "WeekSin", "WeekCos"]
]

RESIDUAL_CATEGORICAL_FEATURES = [
    col for col in CATEGORICAL_FEATURES if col in RESIDUAL_FEATURES
]

RESIDUAL_NUMERIC_FEATURES = [
    col for col in NUMERIC_FEATURES if col in RESIDUAL_FEATURES
]

FIXED_WEIGHT_LOG_RF = 0.313
FIXED_WEIGHT_HOLT_RESIDUAL = 0.687


def safe_expm1(x):
    x = np.asarray(x)
    x = np.clip(x, -50, 20)
    return np.maximum(0, np.expm1(x))


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_rf_pipeline(categorical_features, numeric_features, min_samples_leaf=5):
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=min_samples_leaf,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_competitor_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df = df.dropna(
        subset=["Revenue", "FlightWeek", "MainAirlineCarrier"]
    ).copy()
    df = df[
        df["MainAirlineCarrier"].astype(str).str.upper() != "EW"
    ].copy()

    df["WeekNumber"] = (
        df["FlightWeek"].astype(str).str.extract(r"W(\d+)").astype(int)
    )
    df["Route"] = (
        df["OriginAirport"].astype(str)
        + "_"
        + df["DestinationAirport"].astype(str)
    )
    df["CountryRoute"] = (
        df["OriginCountry"].astype(str)
        + "_"
        + df["DestinationCountry"].astype(str)
    )
    df["WeekSin"] = np.sin(2 * np.pi * df["WeekNumber"] / 52)
    df["WeekCos"] = np.cos(2 * np.pi * df["WeekNumber"] / 52)
    df["LogRevenue"] = np.log1p(df["Revenue"])

    for col in ["IsConnectingFlight", "NumberOfNights", "BookingHorizon"]:
        if col in df.columns:
            if col == "IsConnectingFlight":
                s = df[col]
                if s.dtype == bool:
                    df[col] = s.astype(int)
                else:
                    m = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
                    df[col] = (
                        s.astype(str).str.lower().map(m)
                        .fillna(pd.to_numeric(s, errors="coerce"))
                        .fillna(0).astype(int)
                    )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fit_holt_trend_map(train_df: pd.DataFrame) -> dict[int, float]:
    """Weekly mean log revenue trend for weeks 1-52."""
    weeks = sorted(train_df["WeekNumber"].unique())
    weekly_train_log = (
        train_df.groupby("WeekNumber")["LogRevenue"]
        .mean()
        .reindex(weeks)
        .interpolate()
        .ffill()
        .bfill()
    )
    holt_model = ExponentialSmoothing(
        weekly_train_log.values,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    trend_train_log = pd.Series(holt_model.fittedvalues, index=weeks)
    trend_map = {int(w): float(trend_train_log.loc[w]) for w in weeks}

    max_week = max(weeks)
    if max_week < 52:
        extra = holt_model.forecast(52 - max_week)
        for i, val in enumerate(extra, start=max_week + 1):
            trend_map[int(i)] = float(val)

    return trend_map


def train_models(train_df: pd.DataFrame) -> dict:
    log_rf = build_rf_pipeline(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    log_rf.fit(train_df[BASE_FEATURES], train_df["LogRevenue"])

    trend_map = fit_holt_trend_map(train_df)
    train_df = train_df.copy()
    train_df["TrendLogPrediction"] = train_df["WeekNumber"].map(trend_map)
    train_df["ResidualLogRevenue"] = (
        train_df["LogRevenue"] - train_df["TrendLogPrediction"]
    )

    residual_rf = build_rf_pipeline(
        RESIDUAL_CATEGORICAL_FEATURES,
        RESIDUAL_NUMERIC_FEATURES,
    )
    residual_rf.fit(
        train_df[RESIDUAL_FEATURES],
        train_df["ResidualLogRevenue"],
    )

    return {
        "log_rf": log_rf,
        "residual_rf": residual_rf,
        "trend_map": trend_map,
    }


def build_feature_row(
    *,
    week_number: int,
    origin_airport: str,
    destination_airport: str,
    origin_country: str,
    destination_country: str,
    carrier: str,
    cabin_class: str,
    trip_type: str,
    user_country: str,
    booking_horizon: float,
    number_of_nights: float,
    is_connecting: int,
) -> pd.DataFrame:
    route = f"{origin_airport}_{destination_airport}"
    country_route = f"{origin_country}_{destination_country}"
    row = {
        "WeekNumber": week_number,
        "WeekSin": np.sin(2 * np.pi * week_number / 52),
        "WeekCos": np.cos(2 * np.pi * week_number / 52),
        "Route": route,
        "CountryRoute": country_route,
        "OriginAirport": origin_airport,
        "OriginCountry": origin_country,
        "DestinationAirport": destination_airport,
        "DestinationCountry": destination_country,
        "MainAirlineCarrier": carrier,
        "IsConnectingFlight": is_connecting,
        "UserCountry": user_country,
        "TripType": trip_type,
        "CabinClass": cabin_class,
        "NumberOfNights": number_of_nights,
        "BookingHorizon": booking_horizon,
    }
    return pd.DataFrame([row])


def predict_ensemble_row(
    feature_df: pd.DataFrame,
    log_rf,
    residual_rf,
    trend_map: dict[int, float],
) -> dict[str, float]:
    week = int(feature_df["WeekNumber"].iloc[0])
    trend_log = trend_map.get(week, trend_map.get(max(trend_map.keys()), 0.0))

    log_pred = log_rf.predict(feature_df[BASE_FEATURES])
    log_rf_revenue = float(safe_expm1(log_pred)[0])

    residual_log = residual_rf.predict(feature_df[RESIDUAL_FEATURES])
    holt_log = trend_log + float(residual_log[0])
    holt_revenue = float(safe_expm1(holt_log))

    ensemble = (
        FIXED_WEIGHT_LOG_RF * log_rf_revenue
        + FIXED_WEIGHT_HOLT_RESIDUAL * holt_revenue
    )

    return {
        "ensemble_eur": round(ensemble, 2),
        "log_rf_eur": round(log_rf_revenue, 2),
        "holt_residual_eur": round(holt_revenue, 2),
    }


def predict_all_carriers(
    user_input: dict,
    carriers: list[str],
    log_rf,
    residual_rf,
    trend_map: dict[int, float],
) -> pd.DataFrame:
    rows = []
    for carrier in carriers:
        feat = build_feature_row(carrier=carrier, **user_input)
        preds = predict_ensemble_row(feat, log_rf, residual_rf, trend_map)
        rows.append(
            {
                "Carrier": carrier,
                "Ensemble_Price_EUR": preds["ensemble_eur"],
                "LogRF_Price_EUR": preds["log_rf_eur"],
                "HoltResidual_Price_EUR": preds["holt_residual_eur"],
            }
        )
    out = pd.DataFrame(rows).sort_values("Ensemble_Price_EUR").reset_index(drop=True)
    out.index = out.index + 1
    return out


def build_metadata(train_df: pd.DataFrame) -> dict:
    airport_country = (
        train_df.groupby("OriginAirport")["OriginCountry"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .to_dict()
    )
    dest_country = (
        train_df.groupby("DestinationAirport")["DestinationCountry"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        .to_dict()
    )

    return {
        "carriers": sorted(train_df["MainAirlineCarrier"].astype(str).unique()),
        "origin_airports": sorted(train_df["OriginAirport"].astype(str).unique()),
        "destination_airports": sorted(
            train_df["DestinationAirport"].astype(str).unique()
        ),
        "origin_airport_country": airport_country,
        "destination_airport_country": dest_country,
        "cabin_classes": sorted(train_df["CabinClass"].astype(str).unique()),
        "trip_types": sorted(train_df["TripType"].astype(str).unique()),
        "user_countries": sorted(train_df["UserCountry"].astype(str).unique()),
        "weights": {
            "log_rf": FIXED_WEIGHT_LOG_RF,
            "holt_residual": FIXED_WEIGHT_HOLT_RESIDUAL,
        },
    }


def export_models(
    train_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = train_models(train_df)
    joblib.dump(artifacts["log_rf"], output_dir / "log_rf.joblib")
    joblib.dump(artifacts["residual_rf"], output_dir / "residual_rf.joblib")

    with open(output_dir / "trend_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in artifacts["trend_map"].items()},
            f,
            indent=2,
        )

    metadata = build_metadata(train_df)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved models to {output_dir}")


def load_artifacts(models_dir: str | Path) -> tuple:
    models_dir = Path(models_dir)
    log_rf = joblib.load(models_dir / "log_rf.joblib")
    residual_rf = joblib.load(models_dir / "residual_rf.joblib")
    with open(models_dir / "trend_map.json", encoding="utf-8") as f:
        raw = json.load(f)
    trend_map = {int(k): float(v) for k, v in raw.items()}
    with open(models_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return log_rf, residual_rf, trend_map, metadata
