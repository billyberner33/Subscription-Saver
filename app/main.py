from __future__ import annotations

from typing import Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import agent
from . import db


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Subscription Save Offer Agent", lifespan=lifespan)
_BASE_DIR = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    customers = db.list_customers()
    reasons = [
        ("too_expensive", "Too expensive"),
        ("not_using", "Not using it enough"),
        ("missing_features", "Missing features"),
        ("found_alternative", "Found an alternative"),
        ("bug_or_quality", "Bugs / quality issues"),
        ("temporary_need", "Temporary need"),
    ]
    return templates.TemplateResponse(
        request,
        "index.html",
        {"customers": customers, "reasons": reasons},
    )


@app.get("/api/customers")
def customers() -> JSONResponse:
    return JSONResponse({"customers": db.list_customers(), "note": "Mock customer database (SQLite)."})


@app.get("/api/customer/{customer_id}/loyalty")
def loyalty(customer_id: str, cancellation_reason: str | None = None) -> JSONResponse:
    row = db.get_loyalty(customer_id)
    if not row:
        return JSONResponse({"error": "unknown_customer", "customer_id": customer_id}, status_code=404)
    return JSONResponse(
        {
            "loyalty": {**row, **db.cancellation_summary(customer_id)},
            "cancellation_events": db.list_cancellation_events(customer_id, limit=10),
            "unsubscribe_estimate": db.estimate_unsubscribe_risk(customer_id, cancellation_reason=cancellation_reason),
            "note": "Mocked loyalty metrics + cancellation history (SQLite).",
        }
    )


@app.post("/api/recommend")
async def recommend(request: Request) -> JSONResponse:
    body = await request.json()
    customer_id = str(body.get("customer_id", "")).strip()
    cancellation_reason = str(body.get("cancellation_reason", "")).strip()

    if not customer_id or not cancellation_reason:
        return JSONResponse({"error": "customer_id and cancellation_reason are required"}, status_code=400)

    result = agent.recommend_offer(customer_id=customer_id, cancellation_reason=cancellation_reason)  # type: ignore[arg-type]

    return JSONResponse(
        {
            "status": result.status,
            "decision": result.decision,
            "audit_log": result.audit_log,
            "error": result.error,
        }
    )


@app.post("/api/apply")
async def apply_offer(request: Request) -> JSONResponse:
    body = await request.json()
    selected_offer_id = str(body.get("selected_offer_id", "")).strip()
    customer_id = str(body.get("customer_id", "")).strip()
    if not selected_offer_id or not customer_id:
        return JSONResponse({"error": "selected_offer_id and customer_id are required"}, status_code=400)
    # Mocked “apply” action for the prototype.
    applied: dict[str, Any] = {"customer_id": customer_id, "selected_offer_id": selected_offer_id}
    if selected_offer_id == "offer_custom_discount":
        applied["discount_percent"] = int(body.get("discount_percent") or 0)
        applied["discount_duration_months"] = int(body.get("discount_duration_months") or 0)
    return JSONResponse({"ok": True, "applied": applied})
