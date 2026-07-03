"""
OpenAI pricing analyst — initial verdict + follow-up chat on forecast data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

BA_SYSTEM_PROMPT = """You are a senior airline pricing business analyst at Eurowings (EW).

Rules:
- Use ONLY the JSON forecast data provided. Never invent prices, carriers, or market facts.
- EW is NOT in the competitor list — you advise what EW should DO with a planned or current EW fare.
- Write like an internal Slack note to revenue management: direct, numbers-first, zero fluff.
- Be precise and short. Every sentence must help a decision.

Initial analysis format (use exactly these headings):
## Verdict
One sentence: what EW should do on this route/segment (e.g. "Undercut W6 by €5–€10" or "Hold premium — gap is narrow").

## Do now
Exactly 3 numbered actions. Each must include a € figure from the data (cheapest, median, or spread).
Example: "1. Launch Business fare at €155–€165 to sit €5 below W6 (€159.71)."

## Risk
One bullet: the main downside if EW follows this plan.

## If EW asks you later
Stay in analyst mode. Answer follow-ups in 2–5 short sentences or bullets.
If they propose an idea (e.g. "we want €175"), stress-test it against the forecast numbers.
Ask clarifying questions only when essential (e.g. unknown EW current price — ask once, then give conditional advice).
Never repeat the full initial analysis unless asked."""


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
    sorted_carriers = [
        {"carrier": row["Carrier"], "price_eur": float(row["Ensemble_Price_EUR"])}
        for _, row in results.iterrows()
    ]

    return {
        "search": {
            "week": f"2024-W{user_input['week_number']:02d}",
            "route": f"{user_input['origin_airport']} → {user_input['destination_airport']}",
            "countries": f"{user_input['origin_country']} → {user_input['destination_country']}",
            "cabin": user_input["cabin_class"],
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
            "price_spread_eur": round(float(prices.max() - prices.min()), 2),
            "median_price_eur": round(float(prices.median()), 2),
            "mean_price_eur": round(float(prices.mean()), 2),
            "carrier_count": len(results),
        },
        "all_competitor_prices_sorted": sorted_carriers,
    }


def context_block(context: dict) -> str:
    return f"Forecast data (source of truth):\n```json\n{json.dumps(context, indent=2)}\n```"


def rule_based_summary(context: dict) -> str:
    s = context["search"]
    m = context["market_summary"]
    target_low = round(m["cheapest_price_eur"] - 5, 2)
    target_high = round(m["cheapest_price_eur"], 2)
    return f"""## Verdict
Match or slightly undercut **{m['cheapest_carrier']}** (€{m['cheapest_price_eur']:.2f}) on {s['route']}, {s['cabin']}, {s['week']}.

## Do now
1. Set EW fare at **€{target_low:.2f}–€{target_high:.2f}** to compete with {m['cheapest_carrier']}.
2. If EW is above **€{m['median_price_eur']:.2f}** (median), run a limited promo — spread is €{m['price_spread_eur']:.2f}.
3. Monitor {m['most_expensive_carrier']} (€{m['most_expensive_price_eur']:.2f}) — premium anchor only if EW product justifies it.

## Risk
Undercutting too far on a €{m['price_spread_eur']:.2f} spread erodes margin without gaining share if {m['cheapest_carrier']} matches."""


def _call_openai(api_key: str, messages: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.25,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def generate_initial_analysis(context: dict, api_key: str) -> str:
    messages = [
        {"role": "system", "content": BA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Give your initial pricing recommendation for Eurowings based on this forecast.\n\n"
                + context_block(context)
            ),
        },
    ]
    return _call_openai(api_key, messages)


def chat_followup(
    context: dict,
    chat_history: list[dict],
    user_message: str,
    api_key: str,
) -> str:
    messages = [
        {"role": "system", "content": BA_SYSTEM_PROMPT + "\n\n" + context_block(context)},
    ]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return _call_openai(api_key, messages)


def get_initial_insights(context: dict) -> tuple[str, str, list[dict]]:
    """
    Returns (summary_text, source, chat_history) where chat_history starts with assistant verdict.
    """
    api_key = get_openai_api_key()

    if api_key:
        try:
            summary = generate_initial_analysis(context, api_key)
            history = [{"role": "assistant", "content": summary}]
            return summary, "llm", history
        except Exception as exc:
            summary = rule_based_summary(context)
            summary += f"\n\n*AI unavailable ({exc}). Rule-based verdict shown.*"
            return summary, "rules", [{"role": "assistant", "content": summary}]

    summary = rule_based_summary(context)
    return summary, "rules", [{"role": "assistant", "content": summary}]


def get_chat_reply(
    context: dict,
    chat_history: list[dict],
    user_message: str,
) -> tuple[str, str]:
    api_key = get_openai_api_key()
    if not api_key:
        reply = (
            "Set **OPENAI_API_KEY** in `.env` to chat with the analyst. "
            f"Your idea noted: “{user_message}”. "
            f"Cheapest competitor is {context['market_summary']['cheapest_carrier']} "
            f"at €{context['market_summary']['cheapest_price_eur']:.2f}."
        )
        return reply, "rules"

    try:
        reply = chat_followup(context, chat_history, user_message, api_key)
        return reply, "llm"
    except Exception as exc:
        return f"Could not reach the analyst ({exc}). Try again.", "rules"
