"""
OpenAI pricing analyst — short market summary + follow-up chat.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# From final.ipynb feature importance (competitor price models)
MODEL_DRIVERS = {
    "top_drivers": [
        "CabinClass — strongest price driver across models",
        "Week / seasonality (WeekCos) — fares shift through the year",
        "MainAirlineCarrier — each airline prices differently on the same route",
        "BookingHorizon — longer lead time often means different fare level",
        "Route and airport pair — origin/destination market matters",
    ],
    "cabin_notes": {
        "Economy": "Most price-sensitive segment; small € moves change competitiveness.",
        "Premium Economy": "Mid-tier; cabin and week explain most of the gap vs Economy.",
        "Business": "Wider spread between carriers; premium service can justify above median.",
    },
}

INITIAL_PROMPT = """You are a Eurowings pricing analyst. Use ONLY the JSON data.

Write exactly 4 short paragraphs (2 sentences max each). Plain text — NO headings, NO bullets, NO "if you ask later", NO rhetorical questions.

Paragraph 1 — Market: lowest competitor price (€ + carrier), highest (€ + carrier), and average competitor price. Mention week, route, cabin.

Paragraph 2 — Drivers: one sentence on what mainly drives price in this segment (use model_drivers from JSON).

Paragraph 3 — EW fare: one clear recommended € price or narrow range for Eurowings on this search.

Paragraph 4 — Quality: one sentence that EW can win bookings on service/network even when not the cheapest.

Stay under 120 words total. Be direct like a business analyst memo."""

CHAT_PROMPT = """You are a Eurowings pricing analyst in a follow-up chat.

Use ONLY the forecast JSON. The user already saw an initial recommendation — answer their new question in 2–4 short sentences.
Give a clear yes/no or € figure when they propose a price. No headings. No repeating the full market summary unless they ask."""


def get_openai_api_key() -> str | None:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip() or None
    except Exception:
        pass

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key

    env_path = ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
            return os.getenv("OPENAI_API_KEY", "").strip() or None
        except ImportError:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY"):
                    _, _, value = line.partition("=")
                    return value.strip().strip('"').strip("'") or None

    return None


def build_context(user_input: dict, results: pd.DataFrame) -> dict:
    prices = results["Ensemble_Price_EUR"]
    cheapest = results.iloc[0]
    priciest = results.iloc[-1]
    cabin = user_input["cabin_class"]

    return {
        "search": {
            "week": f"2024-W{user_input['week_number']:02d}",
            "route": f"{user_input['origin_airport']} → {user_input['destination_airport']}",
            "countries": f"{user_input['origin_country']} → {user_input['destination_country']}",
            "cabin": cabin,
            "trip_type": user_input["trip_type"],
            "user_country": user_input["user_country"],
            "booking_horizon_days": user_input["booking_horizon"],
            "nights": user_input["number_of_nights"],
            "connecting": bool(user_input["is_connecting"]),
        },
        "market_summary": {
            "cheapest_carrier": cheapest["Carrier"],
            "cheapest_price_eur": float(cheapest["Ensemble_Price_EUR"]),
            "most_expensive_carrier": priciest["Carrier"],
            "most_expensive_price_eur": float(priciest["Ensemble_Price_EUR"]),
            "average_price_eur": round(float(prices.mean()), 2),
            "median_price_eur": round(float(prices.median()), 2),
            "price_spread_eur": round(float(prices.max() - prices.min()), 2),
            "carrier_count": len(results),
        },
        "all_competitor_prices_sorted": [
            {"carrier": row["Carrier"], "price_eur": float(row["Ensemble_Price_EUR"])}
            for _, row in results.iterrows()
        ],
        "model_drivers": MODEL_DRIVERS,
        "segment_note": MODEL_DRIVERS["cabin_notes"].get(
            cabin, "Cabin and week are the main levers on this route."
        ),
    }


def format_market_snapshot(context: dict) -> str:
    s = context["search"]
    m = context["market_summary"]
    return (
        f"**{s['route']}** · {s['cabin']} · {s['week']} · {s['trip_type']}\n\n"
        f"| | |\n|---|---|\n"
        f"| **Lowest** | {m['cheapest_carrier']} — **€{m['cheapest_price_eur']:.2f}** |\n"
        f"| **Highest** | {m['most_expensive_carrier']} — **€{m['most_expensive_price_eur']:.2f}** |\n"
        f"| **Average** | **€{m['average_price_eur']:.2f}** (median €{m['median_price_eur']:.2f}) |\n"
        f"| **Spread** | €{m['price_spread_eur']:.2f} across {m['carrier_count']} carriers |"
    )


def rule_based_recommendation(context: dict) -> str:
    s = context["search"]
    m = context["market_summary"]
    ew_price = round(m["cheapest_price_eur"] - 5, 2)
    return (
        f"Competitors on {s['route']} in {s['week']} range from "
        f"€{m['cheapest_price_eur']:.2f} ({m['cheapest_carrier']}) to "
        f"€{m['most_expensive_price_eur']:.2f} ({m['most_expensive_carrier']}); "
        f"the market average is €{m['average_price_eur']:.2f}. "
        f"{context['segment_note']} "
        f"A sensible EW {s['cabin']} fare is around **€{ew_price:.2f}–€{m['cheapest_price_eur']:.2f}** "
        f"to stay competitive. EW can still capture demand above the cheapest fare when "
        f"customers value schedule, direct routes, or service quality."
    )


def context_block(context: dict) -> str:
    return f"Forecast data:\n```json\n{json.dumps(context, indent=2)}\n```"


def _call_openai(api_key: str, messages: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def generate_initial_recommendation(context: dict, api_key: str) -> str:
    messages = [
        {"role": "system", "content": INITIAL_PROMPT},
        {
            "role": "user",
            "content": "Write the 4-paragraph analyst note.\n\n" + context_block(context),
        },
    ]
    return _call_openai(api_key, messages)


def chat_followup(
    context: dict,
    initial_recommendation: str,
    chat_history: list[dict],
    user_message: str,
    api_key: str,
) -> str:
    system = (
        CHAT_PROMPT
        + "\n\nInitial recommendation already shown:\n"
        + initial_recommendation
        + "\n\n"
        + context_block(context)
    )
    messages = [{"role": "system", "content": system}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return _call_openai(api_key, messages)


def get_initial_insights(context: dict) -> tuple[str, str]:
    """Returns (recommendation_text, source). Chat history starts empty."""
    api_key = get_openai_api_key()

    if api_key:
        try:
            return generate_initial_recommendation(context, api_key), "llm"
        except Exception as exc:
            text = rule_based_recommendation(context)
            return f"{text}\n\n*AI unavailable ({exc}).*", "rules"

    return rule_based_recommendation(context), "rules"


def get_chat_reply(
    context: dict,
    initial_recommendation: str,
    chat_history: list[dict],
    user_message: str,
) -> tuple[str, str]:
    api_key = get_openai_api_key()
    if not api_key:
        m = context["market_summary"]
        return (
            f"Cheapest competitor: {m['cheapest_carrier']} at €{m['cheapest_price_eur']:.2f}. "
            f"Market average: €{m['average_price_eur']:.2f}. "
            f"Add OPENAI_API_KEY for full chat.",
            "rules",
        )

    try:
        reply = chat_followup(
            context, initial_recommendation, chat_history, user_message, api_key
        )
        return reply, "llm"
    except Exception as exc:
        return f"Could not reach analyst ({exc}). Try again.", "rules"
