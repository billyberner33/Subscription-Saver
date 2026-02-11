from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


CancellationReason = Literal[
    "too_expensive",
    "not_using",
    "missing_features",
    "found_alternative",
    "bug_or_quality",
    "temporary_need",
]


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    segment: Literal["consumer", "pro", "team"]
    country: str
    tenure_months: int
    lifetime_value_usd: float
    payment_risk: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Subscription:
    subscription_id: str
    customer_id: str
    plan: Literal["basic", "plus", "pro", "team"]
    cadence: Literal["monthly", "annual"]
    price_usd: float
    renewal_date: date
    seats: int
    status: Literal["active", "past_due", "canceled"]


@dataclass(frozen=True)
class Usage:
    customer_id: str
    last_30d_active_days: int
    last_30d_key_actions: int
    last_30d_value_score: int  # 0-100
    primary_feature: str
    support_tickets_90d: int


def demo_customers() -> list[Customer]:
    return [
        Customer(
            customer_id="cust_1001",
            name="Avery",
            segment="consumer",
            country="US",
            tenure_months=3,
            lifetime_value_usd=45.0,
            payment_risk="low",
        ),
        Customer(
            customer_id="cust_1002",
            name="Jordan",
            segment="pro",
            country="US",
            tenure_months=14,
            lifetime_value_usd=520.0,
            payment_risk="low",
        ),
        Customer(
            customer_id="cust_1003",
            name="Sam",
            segment="team",
            country="DE",
            tenure_months=26,
            lifetime_value_usd=2200.0,
            payment_risk="medium",
        ),
        Customer(
            customer_id="cust_1004",
            name="Morgan",
            segment="pro",
            country="US",
            tenure_months=2,
            lifetime_value_usd=60.0,
            payment_risk="high",
        ),
    ]


def demo_subscriptions(today: date) -> list[Subscription]:
    return [
        Subscription(
            subscription_id="sub_2001",
            customer_id="cust_1001",
            plan="plus",
            cadence="monthly",
            price_usd=15.0,
            renewal_date=today.replace(day=min(today.day, 28)),
            seats=1,
            status="active",
        ),
        Subscription(
            subscription_id="sub_2002",
            customer_id="cust_1002",
            plan="pro",
            cadence="monthly",
            price_usd=29.0,
            renewal_date=today.replace(day=min(today.day, 28)),
            seats=1,
            status="active",
        ),
        Subscription(
            subscription_id="sub_2003",
            customer_id="cust_1003",
            plan="team",
            cadence="annual",
            price_usd=2400.0,
            renewal_date=today.replace(day=min(today.day, 28)),
            seats=10,
            status="active",
        ),
        Subscription(
            subscription_id="sub_2004",
            customer_id="cust_1004",
            plan="pro",
            cadence="monthly",
            price_usd=29.0,
            renewal_date=today.replace(day=min(today.day, 28)),
            seats=1,
            status="past_due",
        ),
    ]


def demo_usage() -> list[Usage]:
    return [
        Usage(
            customer_id="cust_1001",
            last_30d_active_days=3,
            last_30d_key_actions=2,
            last_30d_value_score=18,
            primary_feature="templates",
            support_tickets_90d=0,
        ),
        Usage(
            customer_id="cust_1002",
            last_30d_active_days=18,
            last_30d_key_actions=44,
            last_30d_value_score=72,
            primary_feature="exports",
            support_tickets_90d=1,
        ),
        Usage(
            customer_id="cust_1003",
            last_30d_active_days=24,
            last_30d_key_actions=180,
            last_30d_value_score=84,
            primary_feature="collaboration",
            support_tickets_90d=4,
        ),
        Usage(
            customer_id="cust_1004",
            last_30d_active_days=1,
            last_30d_key_actions=0,
            last_30d_value_score=5,
            primary_feature="unknown",
            support_tickets_90d=2,
        ),
    ]


def offer_catalog() -> list[dict[str, Any]]:
    # Keep this intentionally small and opinionated to make the “agentic” choice legible.
    return [
        {
            "offer_id": "offer_pause_1mo",
            "type": "pause",
            "label": "Pause 1 month",
            "constraints": {"max_payment_risk": "high"},
            "cost": {"discount_percent": 0, "duration_months": 1},
        },
        {
            "offer_id": "offer_pause_2mo",
            "type": "pause",
            "label": "Pause 2 months",
            "constraints": {"max_payment_risk": "medium"},
            "cost": {"discount_percent": 0, "duration_months": 2},
        },
        {
            "offer_id": "offer_20pct_3mo",
            "type": "discount",
            "label": "20% off for 3 months",
            "constraints": {"min_tenure_months": 2, "max_payment_risk": "medium"},
            "cost": {"discount_percent": 20, "duration_months": 3},
        },
        {
            "offer_id": "offer_30pct_2mo",
            "type": "discount",
            "label": "30% off for 2 months",
            "constraints": {"min_tenure_months": 6, "max_payment_risk": "low"},
            "cost": {"discount_percent": 30, "duration_months": 2},
        },
        {
            "offer_id": "offer_custom_discount",
            "type": "custom_discount",
            "label": "Custom discount (agent sized)",
            "constraints": {"min_tenure_months": 1, "max_payment_risk": "medium"},
            "cost": {"discount_percent": None, "duration_months": None},
        },
        {
            "offer_id": "offer_switch_annual_15pct",
            "type": "cadence_change",
            "label": "Switch to annual (15% off)",
            "constraints": {"min_value_score": 50, "max_payment_risk": "medium"},
            "cost": {"discount_percent": 15, "duration_months": 12},
        },
        {
            "offer_id": "offer_downgrade_plus",
            "type": "downgrade",
            "label": "Downgrade to Plus",
            "constraints": {"plan_not": ["basic", "plus"]},
            "cost": {"discount_percent": 0, "duration_months": 0},
        },
    ]


def _risk_rank(risk: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(risk, 2)


def eligible_offers(customer: Customer, subscription: Subscription, usage: Usage) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []
    for offer in offer_catalog():
        c = offer.get("constraints", {})
        if "min_tenure_months" in c and customer.tenure_months < int(c["min_tenure_months"]):
            continue
        if "min_value_score" in c and usage.last_30d_value_score < int(c["min_value_score"]):
            continue
        if "max_payment_risk" in c and _risk_rank(customer.payment_risk) > _risk_rank(str(c["max_payment_risk"])):
            continue
        if "plan_not" in c and subscription.plan in set(c["plan_not"]):
            continue
        offers.append(offer)
    return offers


def simulate_impact(
    *,
    customer: Customer,
    subscription: Subscription,
    usage: Usage,
    cancellation_reason: CancellationReason,
    offer_id: str,
    discount_percent: int | None = None,
    duration_months: int | None = None,
) -> dict[str, Any]:
    # Heuristic simulator: returns directional metrics that the agent can use for tradeoffs.
    # These are mocked numbers; the prototype UI surfaces them as such.
    base_churn_risk = 0.65
    if usage.last_30d_value_score >= 70:
        base_churn_risk -= 0.12
    if customer.tenure_months >= 12:
        base_churn_risk -= 0.08
    if subscription.status == "past_due":
        base_churn_risk += 0.12
    if cancellation_reason == "too_expensive":
        base_churn_risk += 0.08
    if cancellation_reason == "not_using":
        base_churn_risk += 0.10

    offer = next((o for o in offer_catalog() if o["offer_id"] == offer_id), None)
    if not offer:
        return {"error": "unknown_offer"}

    uplift = 0.0
    margin_impact = 0.0
    t = offer["type"]
    if t == "pause":
        uplift = 0.10 if cancellation_reason in ("temporary_need", "not_using") else 0.05
        margin_impact = -0.01
    elif t in ("discount", "custom_discount"):
        if t == "custom_discount":
            if discount_percent is None or duration_months is None:
                return {"error": "missing_discount_params"}
            discount = max(0.0, min(0.95, float(discount_percent) / 100.0))
        else:
            discount = float(offer["cost"]["discount_percent"]) / 100.0
        uplift = 0.18 if cancellation_reason == "too_expensive" else 0.08
        margin_impact = -min(0.25, discount * 0.9)
    elif t == "cadence_change":
        uplift = 0.14 if usage.last_30d_value_score >= 50 else 0.06
        margin_impact = -0.07
    elif t == "downgrade":
        uplift = 0.09 if cancellation_reason in ("too_expensive", "not_using") else 0.05
        margin_impact = -0.03

    # Clamp and provide a few extra “explainable” outputs.
    predicted_churn_risk = max(0.05, min(0.95, base_churn_risk - uplift))
    return {
        "base_churn_risk": round(base_churn_risk, 2),
        "predicted_churn_risk": round(predicted_churn_risk, 2),
        "retention_uplift": round(uplift, 2),
        "margin_impact": round(margin_impact, 2),
        "notes": "Mocked heuristics for prototype (replace with real models/experiments).",
    }
