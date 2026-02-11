from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv

from openai import OpenAI

from . import data as demo
from . import db


_ENV_PATH = (Path(__file__).resolve().parents[1] / ".env").as_posix()
load_dotenv(dotenv_path=_ENV_PATH)


AgentStatus = Literal["ok", "missing_api_key", "error"]


@dataclass(frozen=True)
class AgentResult:
    status: AgentStatus
    decision: dict[str, Any] | None
    audit_log: list[dict[str, Any]]
    error: str | None = None


def _today() -> date:
    return date.today()


def _get_customer(customer_id: str) -> dict[str, Any]:
    db.init_db()
    customer = db.get_customer(customer_id)
    if not customer:
        return {"error": "unknown_customer", "customer_id": customer_id}
    return customer


def _get_subscription(customer_id: str) -> dict[str, Any]:
    db.init_db()
    subscription = db.get_subscription(customer_id)
    if not subscription:
        return {"error": "unknown_subscription", "customer_id": customer_id}
    return subscription


def _get_usage(customer_id: str) -> dict[str, Any]:
    db.init_db()
    usage = db.get_usage(customer_id)
    if not usage:
        return {"error": "unknown_usage", "customer_id": customer_id}
    return usage


def _get_eligible_offers(customer_id: str, cancellation_reason: demo.CancellationReason) -> dict[str, Any]:
    _ = cancellation_reason
    db.init_db()
    customer = db.get_customer(customer_id)
    subscription = db.get_subscription(customer_id)
    usage = db.get_usage(customer_id)
    loyalty = db.get_loyalty(customer_id)
    if not customer or not subscription or not usage or not loyalty:
        return {
            "error": "missing_context",
            "customer": bool(customer),
            "subscription": bool(subscription),
            "usage": bool(usage),
            "loyalty": bool(loyalty),
        }

    # Reuse the existing eligibility logic by adapting DB rows into dataclasses.
    customer_obj = demo.Customer(
        customer_id=customer["customer_id"],
        name=customer["name"],
        segment=customer["segment"],
        country=customer["country"],
        tenure_months=int(loyalty["tenure_months"]),
        lifetime_value_usd=float(customer["lifetime_value_usd"]),
        payment_risk=customer["payment_risk"],
    )
    sub_obj = demo.Subscription(
        subscription_id=subscription["subscription_id"],
        customer_id=subscription["customer_id"],
        plan=subscription["plan"],
        cadence=subscription["cadence"],
        price_usd=float(subscription["price_usd"]),
        renewal_date=date.fromisoformat(subscription["renewal_date"]),
        seats=int(subscription["seats"]),
        status=subscription["status"],
    )
    usage_obj = demo.Usage(
        customer_id=usage["customer_id"],
        last_30d_active_days=int(usage["last_30d_active_days"]),
        last_30d_key_actions=int(usage["last_30d_key_actions"]),
        last_30d_value_score=int(usage["last_30d_value_score"]),
        primary_feature=usage["primary_feature"],
        support_tickets_90d=int(usage["support_tickets_90d"]),
    )

    offers = demo.eligible_offers(customer_obj, sub_obj, usage_obj)
    guardrails = _risk_guardrails(customer_id)
    max_discount_percent = int(guardrails.get("max_discount_percent", 0))
    filtered: list[dict[str, Any]] = []
    for offer in offers:
        if offer.get("type") == "discount":
            dp = int(offer.get("cost", {}).get("discount_percent", 0))
            if dp > max_discount_percent:
                continue
        filtered.append(offer)

    return {
        "offers": filtered,
        "guardrails": guardrails,
        "note": "Mocked eligibility rules for prototype (derived from DB context + guardrails).",
    }


def _simulate_offer_impact(
    customer_id: str,
    cancellation_reason: demo.CancellationReason,
    offer_id: str,
    discount_percent: int | None = None,
    duration_months: int | None = None,
) -> dict[str, Any]:
    db.init_db()
    customer = db.get_customer(customer_id)
    subscription = db.get_subscription(customer_id)
    usage = db.get_usage(customer_id)
    loyalty = db.get_loyalty(customer_id)
    if not customer or not subscription or not usage or not loyalty:
        return {"error": "missing_context"}

    # Fold loyalty into a slightly more realistic simulator for the demo.
    loyalty_boost = (int(loyalty["loyalty_score"]) - 50) / 1000.0  # [-0.05, +0.05]
    base = demo.simulate_impact(
        customer=demo.Customer(
            customer_id=customer["customer_id"],
            name=customer["name"],
            segment=customer["segment"],
            country=customer["country"],
            tenure_months=int(loyalty["tenure_months"]),
            lifetime_value_usd=float(customer["lifetime_value_usd"]),
            payment_risk=customer["payment_risk"],
        ),
        subscription=demo.Subscription(
            subscription_id=subscription["subscription_id"],
            customer_id=subscription["customer_id"],
            plan=subscription["plan"],
            cadence=subscription["cadence"],
            price_usd=float(subscription["price_usd"]),
            renewal_date=date.fromisoformat(subscription["renewal_date"]),
            seats=int(subscription["seats"]),
            status=subscription["status"],
        ),
        usage=demo.Usage(
            customer_id=usage["customer_id"],
            last_30d_active_days=int(usage["last_30d_active_days"]),
            last_30d_key_actions=int(usage["last_30d_key_actions"]),
            last_30d_value_score=int(usage["last_30d_value_score"]),
            primary_feature=usage["primary_feature"],
            support_tickets_90d=int(usage["support_tickets_90d"]),
        ),
        cancellation_reason=cancellation_reason,
        offer_id=offer_id,
        discount_percent=discount_percent,
        duration_months=duration_months,
    )
    if "error" in base:
        return base
    base["base_churn_risk"] = round(max(0.05, min(0.95, float(base["base_churn_risk"]) - loyalty_boost)), 2)
    base["predicted_churn_risk"] = round(
        max(0.05, min(0.95, float(base["predicted_churn_risk"]) - loyalty_boost)), 2
    )
    base["loyalty_score"] = int(loyalty["loyalty_score"])
    base["notes"] = str(base.get("notes", "")).rstrip(".") + "; adjusted by loyalty_score (mocked)."
    return base


def _risk_guardrails(customer_id: str) -> dict[str, Any]:
    db.init_db()
    customer = db.get_customer(customer_id)
    loyalty = db.get_loyalty(customer_id)
    if not customer or not loyalty:
        return {"error": "unknown_customer", "customer_id": customer_id}

    # Simplified guardrails for prototype; in real life these come from finance/legal/policy.
    payment_risk = str(customer["payment_risk"])
    max_discount = 0.30 if payment_risk == "low" else 0.20 if payment_risk == "medium" else 0.0

    # If discount sensitivity is high, prefer non-discount levers.
    if int(loyalty["discount_sensitivity"]) >= 80:
        max_discount = min(max_discount, 0.20)

    # If loyalty is high and engagement is strong, avoid large discounts.
    loyalty_score = int(loyalty["loyalty_score"])
    avg_daily_minutes = float(loyalty["avg_daily_minutes"])
    if loyalty_score >= 65 or avg_daily_minutes >= 20:
        max_discount = min(max_discount, 0.15)

    preferred_levers: list[str] = ["pause", "downgrade", "cadence_change"]
    if max_discount > 0:
        preferred_levers.append("discount")

    return {
        "max_discount_percent": int(max_discount * 100),
        "preferred_levers": preferred_levers,
        "notes": "Mocked guardrails from payment risk + loyalty/engagement + discount sensitivity (prototype).",
    }


def _get_loyalty_profile(customer_id: str) -> dict[str, Any]:
    db.init_db()
    loyalty = db.get_loyalty(customer_id)
    if not loyalty:
        return {"error": "unknown_customer", "customer_id": customer_id}
    summary = db.cancellation_summary(customer_id)
    return {**loyalty, **summary}


def _estimate_unsubscribe_risk(customer_id: str, cancellation_reason: demo.CancellationReason) -> dict[str, Any]:
    db.init_db()
    return db.estimate_unsubscribe_risk(customer_id, cancellation_reason=cancellation_reason)


def _tool_definitions() -> list[dict[str, Any]]:
    # Tool schemas are intentionally tight to encourage deterministic flows in the demo.
    return [
        {
            "type": "function",
            "function": {
                "name": "get_customer",
                "description": "Fetch customer profile and segment details.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_subscription",
                "description": "Fetch active subscription plan, cadence, price, renewal date and status.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_usage",
                "description": "Fetch recent usage signals used to tailor the save offer.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_eligible_offers",
                "description": "Return the offer catalog filtered by eligibility rules.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "cancellation_reason": {
                            "type": "string",
                            "enum": [
                                "too_expensive",
                                "not_using",
                                "missing_features",
                                "found_alternative",
                                "bug_or_quality",
                                "temporary_need",
                            ],
                        },
                    },
                    "required": ["customer_id", "cancellation_reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "simulate_offer_impact",
                "description": "Estimate retention uplift and margin impact for a specific offer (mocked).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "cancellation_reason": {
                            "type": "string",
                            "enum": [
                                "too_expensive",
                                "not_using",
                                "missing_features",
                                "found_alternative",
                                "bug_or_quality",
                                "temporary_need",
                            ],
                        },
                        "offer_id": {"type": "string"},
                        "discount_percent": {"type": "integer", "minimum": 0, "maximum": 90},
                        "duration_months": {"type": "integer", "minimum": 1, "maximum": 12},
                    },
                    "required": ["customer_id", "cancellation_reason", "offer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "risk_guardrails",
                "description": "Return policy/finance guardrails the offer must respect (mocked).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_loyalty_profile",
                "description": "Fetch loyalty metrics (tenure, avg daily minutes, total spend, cancel attempts, last cancel date, cancel reason history, NPS, discount sensitivity, loyalty score).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"customer_id": {"type": "string"}},
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_unsubscribe_risk",
                "description": "Estimate unsubscribe likelihood using a simple heuristic and explainable drivers (mocked).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "cancellation_reason": {
                            "type": "string",
                            "enum": [
                                "too_expensive",
                                "not_using",
                                "missing_features",
                                "found_alternative",
                                "bug_or_quality",
                                "temporary_need",
                            ],
                        },
                    },
                    "required": ["customer_id", "cancellation_reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_offer",
                "description": "Finalize the recommended save offer with reasoning and a customer-facing message.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_offer_id": {"type": "string"},
                        "discount_percent": {"type": "integer", "minimum": 0, "maximum": 90},
                        "discount_duration_months": {"type": "integer", "minimum": 1, "maximum": 12},
                        "why_this_offer": {"type": "string"},
                        "tradeoffs": {"type": "array", "items": {"type": "string"}},
                        "alternatives_considered": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                        "expected_outcomes": {
                            "type": "object",
                            "properties": {
                                "retention_uplift": {"type": "number"},
                                "margin_impact": {"type": "number"},
                                "predicted_churn_risk": {"type": "number"},
                            },
                            "required": ["retention_uplift", "margin_impact", "predicted_churn_risk"],
                            "additionalProperties": False,
                        },
                        "customer_message": {"type": "string"},
                        "agent_notes": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": [
                        "selected_offer_id",
                        "why_this_offer",
                        "tradeoffs",
                        "alternatives_considered",
                        "risk_flags",
                        "expected_outcomes",
                        "customer_message",
                        "agent_notes",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _tool_router() -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        "get_customer": lambda *, customer_id: _get_customer(customer_id),
        "get_subscription": lambda *, customer_id: _get_subscription(customer_id),
        "get_usage": lambda *, customer_id: _get_usage(customer_id),
        "get_eligible_offers": lambda *, customer_id, cancellation_reason: _get_eligible_offers(
            customer_id, cancellation_reason
        ),
        "simulate_offer_impact": lambda *, customer_id, cancellation_reason, offer_id, discount_percent=None, duration_months=None: _simulate_offer_impact(
            customer_id,
            cancellation_reason,
            offer_id,
            discount_percent=discount_percent,
            duration_months=duration_months,
        ),
        "risk_guardrails": lambda *, customer_id: _risk_guardrails(customer_id),
        "get_loyalty_profile": lambda *, customer_id: _get_loyalty_profile(customer_id),
        "estimate_unsubscribe_risk": lambda *, customer_id, cancellation_reason: _estimate_unsubscribe_risk(
            customer_id, cancellation_reason
        ),
        "finalize_offer": lambda **kwargs: {"received": True, "final": kwargs},
    }


def recommend_offer(*, customer_id: str, cancellation_reason: demo.CancellationReason) -> AgentResult:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return AgentResult(status="missing_api_key", decision=None, audit_log=[], error="OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "o4-mini").strip() or "o4-mini"

    tools = _tool_definitions()
    router = _tool_router()
    audit_log: list[dict[str, Any]] = []

    system_prompt = (
        "You are a retention decision agent for a subscription business. "
        "Your goal is to recommend the smallest effective save offer that maximizes retention "
        "while respecting guardrails (e.g., max discount, eligibility) and minimizing margin impact.\n\n"
        "Rules:\n"
        "- Use tools to gather facts; do not guess.\n"
        "- Always consider loyalty metrics (tenure, engagement, NPS, discount sensitivity).\n"
        "- Use a larger discount only when loyalty is low AND unsubscribe risk is high, and only within guardrails.\n"
        "- If loyalty/engagement is high, prefer non-discount levers (pause/downgrade/annual) and smaller discounts.\n"
        "- If you choose offer_custom_discount, you MUST set discount_percent and discount_duration_months.\n"
        "- Never exceed max_discount_percent from risk_guardrails.\n"
        "- Only choose from eligible offers.\n"
        "- Prefer: pause/downgrade before discounts unless price is the dominant reason.\n"
        "- If payment risk is high, avoid discounts.\n"
        "- Always call finalize_offer as the last step. Do not output other text.\n"
    )

    user_prompt = {
        "customer_id": customer_id,
        "cancellation_reason": cancellation_reason,
        "task": "Recommend a save offer and provide a customer message + internal notes.",
    }

    def _create_response(
        *,
        previous_response_id: str | None,
        input_items: str | list[dict[str, Any]],
        instructions: str | None = None,
    ):
        base_kwargs: dict[str, Any] = {"model": model, "tools": tools, "input": input_items}
        if instructions:
            base_kwargs["instructions"] = instructions
        if previous_response_id:
            base_kwargs["previous_response_id"] = previous_response_id

        # Prefer deterministic tool plans in a demo (and simpler orchestration).
        attempts: list[dict[str, Any]] = [
            {**base_kwargs, "parallel_tool_calls": False, "reasoning": {"summary": "auto"}},
            {**base_kwargs, "parallel_tool_calls": False},
            {**base_kwargs},
        ]
        last_err: Exception | None = None
        for kwargs in attempts:
            try:
                return client.responses.create(**kwargs)
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("failed to create response")

    try:
        response = _create_response(
            previous_response_id=None,
            instructions=system_prompt,
            input_items=json.dumps(user_prompt),
        )

        final_decision: dict[str, Any] | None = None

        for _ in range(12):
            function_calls = [
                item for item in getattr(response, "output", []) if getattr(item, "type", None) == "function_call"
            ]
            if not function_calls:
                break

            tool_outputs: list[dict[str, Any]] = []
            for call in function_calls:
                name = call.name
                raw_args = getattr(call, "arguments", None) or "{}"
                args = json.loads(raw_args)
                audit_log.append({"tool": name, "args": args})

                handler = router.get(name)
                if not handler:
                    result = {"error": f"unknown_tool:{name}"}
                else:
                    result = handler(**args)

                if name == "finalize_offer" and isinstance(result, dict) and "final" in result:
                    final_decision = result["final"]

                    # Server-side validation/guardrails (do not allow the model to exceed limits).
                    guard = _risk_guardrails(customer_id)
                    max_discount_percent = int(guard.get("max_discount_percent", 0))
                    if final_decision.get("selected_offer_id") == "offer_custom_discount":
                        dp = int(final_decision.get("discount_percent") or 0)
                        dm = int(final_decision.get("discount_duration_months") or 0)
                        if dm <= 0:
                            dm = 2
                            final_decision["discount_duration_months"] = dm
                            final_decision.setdefault("risk_flags", []).append("missing_duration_defaulted")
                        if dp > max_discount_percent:
                            final_decision["discount_percent"] = max_discount_percent
                            final_decision.setdefault("risk_flags", []).append("discount_clamped_to_guardrail")
                        if max_discount_percent == 0:
                            final_decision.setdefault("risk_flags", []).append("discount_not_allowed_by_guardrails")
                            final_decision["confidence"] = "low"

                audit_log.append({"tool": name, "result": result})
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result),
                    }
                )

            response = _create_response(previous_response_id=response.id, input_items=tool_outputs)
            if final_decision:
                return AgentResult(status="ok", decision=final_decision, audit_log=audit_log)

        return AgentResult(status="error", decision=None, audit_log=audit_log, error="Agent did not finalize a decision")
    except Exception as e:
        return AgentResult(status="error", decision=None, audit_log=audit_log, error=str(e))
