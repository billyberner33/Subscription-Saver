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
    customer = next((c for c in demo.demo_customers() if c.customer_id == customer_id), None)
    if not customer:
        return {"error": "unknown_customer", "customer_id": customer_id}
    return customer.__dict__


def _get_subscription(customer_id: str) -> dict[str, Any]:
    subscription = next((s for s in demo.demo_subscriptions(_today()) if s.customer_id == customer_id), None)
    if not subscription:
        return {"error": "unknown_subscription", "customer_id": customer_id}
    payload = subscription.__dict__.copy()
    payload["renewal_date"] = payload["renewal_date"].isoformat()
    return payload


def _get_usage(customer_id: str) -> dict[str, Any]:
    usage = next((u for u in demo.demo_usage() if u.customer_id == customer_id), None)
    if not usage:
        return {"error": "unknown_usage", "customer_id": customer_id}
    return usage.__dict__


def _get_eligible_offers(customer_id: str, cancellation_reason: demo.CancellationReason) -> dict[str, Any]:
    customer = next((c for c in demo.demo_customers() if c.customer_id == customer_id), None)
    subscription = next((s for s in demo.demo_subscriptions(_today()) if s.customer_id == customer_id), None)
    usage = next((u for u in demo.demo_usage() if u.customer_id == customer_id), None)
    if not customer or not subscription or not usage:
        return {"error": "missing_context", "customer": bool(customer), "subscription": bool(subscription), "usage": bool(usage)}
    offers = demo.eligible_offers(customer, subscription, usage)
    return {"offers": offers, "note": "Mocked eligibility rules for prototype."}


def _simulate_offer_impact(
    customer_id: str,
    cancellation_reason: demo.CancellationReason,
    offer_id: str,
) -> dict[str, Any]:
    customer = next((c for c in demo.demo_customers() if c.customer_id == customer_id), None)
    subscription = next((s for s in demo.demo_subscriptions(_today()) if s.customer_id == customer_id), None)
    usage = next((u for u in demo.demo_usage() if u.customer_id == customer_id), None)
    if not customer or not subscription or not usage:
        return {"error": "missing_context"}
    return demo.simulate_impact(
        customer=customer,
        subscription=subscription,
        usage=usage,
        cancellation_reason=cancellation_reason,
        offer_id=offer_id,
    )


def _risk_guardrails(customer_id: str) -> dict[str, Any]:
    customer = next((c for c in demo.demo_customers() if c.customer_id == customer_id), None)
    if not customer:
        return {"error": "unknown_customer", "customer_id": customer_id}
    # Simplified guardrails for prototype; in real life these come from finance/legal/policy.
    max_discount = 0.30 if customer.payment_risk == "low" else 0.20 if customer.payment_risk == "medium" else 0.0
    return {
        "max_discount_percent": int(max_discount * 100),
        "notes": "Mocked guardrails derived from payment risk (prototype).",
    }


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
                "name": "finalize_offer",
                "description": "Finalize the recommended save offer with reasoning and a customer-facing message.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_offer_id": {"type": "string"},
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
        "simulate_offer_impact": lambda *, customer_id, cancellation_reason, offer_id: _simulate_offer_impact(
            customer_id, cancellation_reason, offer_id
        ),
        "risk_guardrails": lambda *, customer_id: _risk_guardrails(customer_id),
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
                item
                for item in getattr(response, "output", [])
                if getattr(item, "type", None) in ("function_call", "custom_tool_call")
            ]
            if not function_calls:
                break

            tool_outputs: list[dict[str, Any]] = []
            for call in function_calls:
                name = call.name
                raw_args = getattr(call, "arguments", None) or getattr(call, "input", None) or "{}"
                args = json.loads(raw_args)
                audit_log.append({"tool": name, "args": args})

                handler = router.get(name)
                if not handler:
                    result = {"error": f"unknown_tool:{name}"}
                else:
                    result = handler(**args)

                if name == "finalize_offer" and isinstance(result, dict) and "final" in result:
                    final_decision = result["final"]

                audit_log.append({"tool": name, "result": result})
                tool_outputs.append(
                    {
                        "type": "custom_tool_call_output",
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
