"""
Eurowings Case Study — Competitor price predictor (Streamlit demo).
Enter trip details once → predicted price for all competitor carriers.
"""

from pathlib import Path

import streamlit as st

from model_utils import (
    load_artifacts,
    load_competitor_data,
    predict_all_carriers,
    train_models,
)

st.set_page_config(
    page_title="Eurowings Competitor Price Forecast",
    page_icon="✈️",
    layout="wide",
)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "skyscanner_airfare_data.csv"


@st.cache_resource
def load_models():
    """Use pre-exported joblib locally, or train once from CSV on Streamlit Cloud."""
    if (MODELS_DIR / "log_rf.joblib").exists():
        return load_artifacts(MODELS_DIR)

    if not DATA_PATH.exists():
        return None

    train_df = load_competitor_data(DATA_PATH)
    artifacts = train_models(train_df)
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        import json

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        from model_utils import build_metadata

        metadata = build_metadata(train_df)

    return (
        artifacts["log_rf"],
        artifacts["residual_rf"],
        artifacts["trend_map"],
        metadata,
    )


def main():
    st.title("✈️ Competitor Airfare Price Forecast")
    st.caption(
        "Case Study 1 demo — ensemble model (31.3% Log RF + 68.7% Holt+Residual). "
        "Enter search details once to compare all competitor carriers."
    )

    with st.spinner("Loading models (first visit may take ~1–2 min)..."):
        artifacts = load_models()
    if artifacts is None:
        st.error(
            "Could not load data. Ensure `skyscanner_airfare_data.csv` is in the repo root."
        )
        st.stop()

    log_rf, residual_rf, trend_map, metadata = artifacts
    carriers = metadata["carriers"]
    origin_map = metadata["origin_airport_country"]
    dest_map = metadata["destination_airport_country"]

    col1, col2, col3 = st.columns(3)

    with col1:
        week_number = st.slider("Flight week", 1, 52, 45)
        origin = st.selectbox("Origin airport", metadata["origin_airports"], index=0)
        destination = st.selectbox(
            "Destination airport",
            metadata["destination_airports"],
            index=min(1, len(metadata["destination_airports"]) - 1),
        )

    with col2:
        cabin = st.selectbox("Cabin class", metadata["cabin_classes"])
        trip_type = st.selectbox("Trip type", metadata["trip_types"])
        user_country = st.selectbox("User country", metadata["user_countries"])

    with col3:
        booking_horizon = st.number_input(
            "Booking horizon (days before flight)", min_value=0, max_value=365, value=30
        )
        number_of_nights = st.number_input(
            "Number of nights (0 for one-way)", min_value=0, max_value=30, value=7
        )
        is_connecting = st.selectbox("Connecting flight?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    origin_country = origin_map.get(origin, "DE")
    dest_country = dest_map.get(destination, "ES")

    st.markdown(
        f"**Route:** `{origin}` ({origin_country}) → `{destination}` ({dest_country}) · "
        f"**Week:** `2024-W{week_number:02d}`"
    )

    if st.button("Predict all competitor prices", type="primary"):
        user_input = {
            "week_number": week_number,
            "origin_airport": origin,
            "destination_airport": destination,
            "origin_country": origin_country,
            "destination_country": dest_country,
            "cabin_class": cabin,
            "trip_type": trip_type,
            "user_country": user_country,
            "booking_horizon": float(booking_horizon),
            "number_of_nights": float(number_of_nights),
            "is_connecting": int(is_connecting),
        }

        with st.spinner("Predicting prices for all carriers..."):
            results = predict_all_carriers(
                user_input, carriers, log_rf, residual_rf, trend_map
            )

        cheapest = results.iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric("Cheapest carrier", cheapest["Carrier"])
        m2.metric("Ensemble price (€)", f"{cheapest['Ensemble_Price_EUR']:.2f}")
        m3.metric("Carriers compared", len(results))

        st.subheader("All competitor prices (sorted cheapest first)")
        st.dataframe(
            results.style.format(
                {
                    "Ensemble_Price_EUR": "€ {:.2f}",
                    "LogRF_Price_EUR": "€ {:.2f}",
                    "HoltResidual_Price_EUR": "€ {:.2f}",
                }
            ),
            use_container_width=True,
        )

        chart_df = results.set_index("Carrier")[["Ensemble_Price_EUR"]]
        st.subheader("Price comparison")
        st.bar_chart(chart_df)

        st.info(
            "Prices are model estimates from Skyscanner training data, not live fares. "
            "Ensemble combines Log RF and Holt+Residual RF (weights tuned in pred3.ipynb)."
        )


if __name__ == "__main__":
    main()
