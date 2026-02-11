from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from . import data as demo


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "mock_commerce.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(today: date | None = None) -> None:
    today = today or date.today()
    with _connect() as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS customers (
              customer_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              segment TEXT NOT NULL,
              country TEXT NOT NULL,
              lifetime_value_usd REAL NOT NULL,
              payment_risk TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subscriptions (
              subscription_id TEXT PRIMARY KEY,
              customer_id TEXT NOT NULL,
              plan TEXT NOT NULL,
              cadence TEXT NOT NULL,
              price_usd REAL NOT NULL,
              renewal_date TEXT NOT NULL,
              seats INTEGER NOT NULL,
              status TEXT NOT NULL,
              FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );

            CREATE TABLE IF NOT EXISTS usage (
              customer_id TEXT PRIMARY KEY,
              last_30d_active_days INTEGER NOT NULL,
              last_30d_key_actions INTEGER NOT NULL,
              last_30d_value_score INTEGER NOT NULL,
              primary_feature TEXT NOT NULL,
              support_tickets_90d INTEGER NOT NULL,
              FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );

            -- Loyalty metrics that a support/retention agent would reference.
            CREATE TABLE IF NOT EXISTS loyalty (
              customer_id TEXT PRIMARY KEY,
              tenure_months INTEGER NOT NULL,
              avg_daily_minutes REAL NOT NULL,
              avg_weekly_sessions REAL NOT NULL,
              nps INTEGER NOT NULL,
              refund_count_12m INTEGER NOT NULL,
              cancellation_attempts_12m INTEGER NOT NULL DEFAULT 0,
              total_spend_usd REAL NOT NULL DEFAULT 0,
              discount_sensitivity INTEGER NOT NULL, -- 0-100
              loyalty_score INTEGER NOT NULL,        -- 0-100 (derived, mocked)
              FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );

            -- Cancellation history for “has tried to cancel before?” and “why?”.
            CREATE TABLE IF NOT EXISTS cancellation_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              customer_id TEXT NOT NULL,
              event_date TEXT NOT NULL, -- ISO-8601 date
              reason TEXT NOT NULL,
              outcome TEXT NOT NULL, -- saved | canceled | aborted
              note TEXT,
              FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );
            """
        )
        _migrate(conn)

    seed_demo_data(today=today)


def seed_demo_data(*, today: date) -> None:
    customers = demo.demo_customers()
    subs = demo.demo_subscriptions(today)
    usage = demo.demo_usage()

    # Mock loyalty profiles; keep numbers plausible and easy to explain in an interview.
    loyalty_rows: dict[str, dict[str, Any]] = {
        "cust_1001": {
            "tenure_months": 3,
            "avg_daily_minutes": 6.5,
            "avg_weekly_sessions": 2.0,
            "nps": 7,
            "refund_count_12m": 0,
            "cancellation_attempts_12m": 1,
            "total_spend_usd": 45.0,
            "discount_sensitivity": 80,
        },
        "cust_1002": {
            "tenure_months": 14,
            "avg_daily_minutes": 22.0,
            "avg_weekly_sessions": 5.5,
            "nps": 9,
            "refund_count_12m": 0,
            "cancellation_attempts_12m": 0,
            "total_spend_usd": 520.0,
            "discount_sensitivity": 35,
        },
        "cust_1003": {
            "tenure_months": 26,
            "avg_daily_minutes": 38.0,
            "avg_weekly_sessions": 9.0,
            "nps": 8,
            "refund_count_12m": 1,
            "cancellation_attempts_12m": 1,
            "total_spend_usd": 2200.0,
            "discount_sensitivity": 25,
        },
        "cust_1004": {
            "tenure_months": 2,
            "avg_daily_minutes": 3.0,
            "avg_weekly_sessions": 1.0,
            "nps": 4,
            "refund_count_12m": 1,
            "cancellation_attempts_12m": 2,
            "total_spend_usd": 60.0,
            "discount_sensitivity": 90,
        },
    }

    for customer_id, row in loyalty_rows.items():
        row["loyalty_score"] = compute_loyalty_score(row)

    with _connect() as conn:
        existing = conn.execute("SELECT COUNT(*) AS c FROM customers").fetchone()["c"]
        if existing:
            return

        for c in customers:
            conn.execute(
                """
                INSERT INTO customers (customer_id, name, segment, country, lifetime_value_usd, payment_risk)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    c.customer_id,
                    c.name,
                    c.segment,
                    c.country,
                    c.lifetime_value_usd,
                    c.payment_risk,
                ),
            )

        for s in subs:
            conn.execute(
                """
                INSERT INTO subscriptions (subscription_id, customer_id, plan, cadence, price_usd, renewal_date, seats, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s.subscription_id,
                    s.customer_id,
                    s.plan,
                    s.cadence,
                    s.price_usd,
                    s.renewal_date.isoformat(),
                    s.seats,
                    s.status,
                ),
            )

        for u in usage:
            conn.execute(
                """
                INSERT INTO usage (customer_id, last_30d_active_days, last_30d_key_actions, last_30d_value_score, primary_feature, support_tickets_90d)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    u.customer_id,
                    u.last_30d_active_days,
                    u.last_30d_key_actions,
                    u.last_30d_value_score,
                    u.primary_feature,
                    u.support_tickets_90d,
                ),
            )

        for customer_id, row in loyalty_rows.items():
            conn.execute(
                """
                INSERT INTO loyalty (
                  customer_id,
                  tenure_months,
                  avg_daily_minutes,
                  avg_weekly_sessions,
                  nps,
                  refund_count_12m,
                  cancellation_attempts_12m,
                  total_spend_usd,
                  discount_sensitivity,
                  loyalty_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    customer_id,
                    row["tenure_months"],
                    row["avg_daily_minutes"],
                    row["avg_weekly_sessions"],
                    row["nps"],
                    row["refund_count_12m"],
                    row["cancellation_attempts_12m"],
                    row["total_spend_usd"],
                    row["discount_sensitivity"],
                    row["loyalty_score"],
                ),
            )

        cancellation_events = [
            {
                "customer_id": "cust_1001",
                "event_date": today.replace(day=max(1, min(28, today.day - 12))).isoformat(),
                "reason": "not_using",
                "outcome": "saved",
                "note": "Offered 1-month pause; customer agreed to try again.",
            },
            {
                "customer_id": "cust_1003",
                "event_date": today.replace(day=max(1, min(28, today.day - 40))).isoformat(),
                "reason": "too_expensive",
                "outcome": "saved",
                "note": "Moved to annual with 15% discount after value review.",
            },
            {
                "customer_id": "cust_1004",
                "event_date": today.replace(day=max(1, min(28, today.day - 18))).isoformat(),
                "reason": "bug_or_quality",
                "outcome": "aborted",
                "note": "Customer left before resolution; follow-up recommended.",
            },
            {
                "customer_id": "cust_1004",
                "event_date": today.replace(day=max(1, min(28, today.day - 4))).isoformat(),
                "reason": "too_expensive",
                "outcome": "saved",
                "note": "Agent offered downgrade; customer stayed for now.",
            },
        ]
        for ev in cancellation_events:
            conn.execute(
                """
                INSERT INTO cancellation_events (customer_id, event_date, reason, outcome, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ev["customer_id"], ev["event_date"], ev["reason"], ev["outcome"], ev.get("note")),
            )


def compute_loyalty_score(row: dict[str, Any]) -> int:
    # Simple explainable scoring for the prototype (not “real ML”):
    # tenure + engagement + sentiment + spend - refunds - cancel attempts - discount sensitivity.
    tenure = min(1.0, float(row["tenure_months"]) / 24.0)
    engagement = min(1.0, float(row["avg_daily_minutes"]) / 40.0) * 0.6 + min(
        1.0, float(row["avg_weekly_sessions"]) / 10.0
    ) * 0.4
    sentiment = (max(0, min(10, int(row["nps"]))) / 10.0)
    spend = min(1.0, float(row.get("total_spend_usd", 0.0)) / 2000.0)
    refunds_penalty = min(1.0, float(row["refund_count_12m"]) / 3.0)
    cancel_penalty = min(1.0, float(row.get("cancellation_attempts_12m", 0)) / 3.0)
    discount_penalty = max(0.0, min(1.0, float(row["discount_sensitivity"]) / 100.0))

    score = (
        0.33 * tenure
        + 0.33 * engagement
        + 0.18 * sentiment
        + 0.06 * spend
        - 0.05 * refunds_penalty
        - 0.07 * cancel_penalty
        - 0.03 * discount_penalty
    )
    return int(round(max(0.0, min(1.0, score)) * 100))


def list_customers() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT customer_id, name, segment, country FROM customers ORDER BY customer_id").fetchall()
        return [dict(r) for r in rows]


def get_customer(customer_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None


def get_subscription(customer_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM subscriptions WHERE customer_id = ? ORDER BY renewal_date DESC LIMIT 1", (customer_id,)
        ).fetchone()
        return dict(row) if row else None


def get_usage(customer_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM usage WHERE customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None


def get_loyalty(customer_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM loyalty WHERE customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None


def list_cancellation_events(customer_id: str, limit: int = 10) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT event_date, reason, outcome, note
            FROM cancellation_events
            WHERE customer_id = ?
            ORDER BY event_date DESC, id DESC
            LIMIT ?
            """,
            (customer_id, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]


def cancellation_summary(customer_id: str) -> dict[str, Any]:
    events = list_cancellation_events(customer_id, limit=50)
    last_attempt = events[0]["event_date"] if events else None
    reasons: dict[str, int] = {}
    for ev in events:
        reasons[ev["reason"]] = reasons.get(ev["reason"], 0) + 1
    return {"last_cancellation_attempt_date": last_attempt, "cancellation_reason_history": reasons}


def estimate_unsubscribe_risk(customer_id: str, cancellation_reason: str | None = None) -> dict[str, Any]:
    """
    Mocked churn/unsubscribe likelihood heuristic.

    Returns a probability in [0,1] and an explainable set of drivers.
    This is intentionally simple + auditable for interview demo purposes.
    """
    customer = get_customer(customer_id)
    subscription = get_subscription(customer_id)
    usage = get_usage(customer_id)
    loyalty = get_loyalty(customer_id)
    if not customer or not subscription or not usage or not loyalty:
        return {"error": "missing_context"}

    drivers: list[str] = []
    risk = 0.45

    # Product usage / realized value.
    value_score = int(usage["last_30d_value_score"])
    if value_score < 20:
        risk += 0.18
        drivers.append("very_low_value_realization")
    elif value_score < 50:
        risk += 0.08
        drivers.append("moderate_value_realization")
    else:
        risk -= 0.06
        drivers.append("high_value_realization")

    avg_daily_minutes = float(loyalty["avg_daily_minutes"])
    if avg_daily_minutes < 5:
        risk += 0.12
        drivers.append("low_engagement_minutes")
    elif avg_daily_minutes > 20:
        risk -= 0.06
        drivers.append("high_engagement_minutes")

    # Prior cancellation attempts predict future churn.
    attempts = int(loyalty.get("cancellation_attempts_12m", 0))
    if attempts >= 2:
        risk += 0.14
        drivers.append("multiple_recent_cancel_attempts")
    elif attempts == 1:
        risk += 0.07
        drivers.append("prior_cancel_attempt")

    # Subscription health.
    if str(subscription["status"]) == "past_due":
        risk += 0.12
        drivers.append("past_due")

    # Sentiment.
    nps = int(loyalty["nps"])
    if nps <= 5:
        risk += 0.06
        drivers.append("low_nps")
    elif nps >= 9:
        risk -= 0.03
        drivers.append("high_nps")

    # Loyalty score is a meta-signal.
    loyalty_score = int(loyalty["loyalty_score"])
    if loyalty_score >= 65:
        risk -= 0.12
        drivers.append("high_loyalty_score")
    elif loyalty_score <= 25:
        risk += 0.10
        drivers.append("low_loyalty_score")

    # Reason-specific adjustments.
    if cancellation_reason:
        if cancellation_reason == "too_expensive":
            risk += 0.07
            drivers.append("price_objection")
        elif cancellation_reason == "not_using":
            risk += 0.09
            drivers.append("non_usage")
        elif cancellation_reason == "bug_or_quality":
            risk += 0.05
            drivers.append("quality_issue")
        elif cancellation_reason == "temporary_need":
            risk -= 0.03
            drivers.append("temporary_need")

    risk = max(0.05, min(0.95, risk))
    bucket: str
    if risk >= 0.75:
        bucket = "high"
    elif risk >= 0.55:
        bucket = "medium"
    else:
        bucket = "low"

    return {
        "unsubscribe_risk": round(risk, 2),
        "bucket": bucket,
        "drivers": drivers,
        "notes": "Mocked heuristic for prototype; replace with churn model + experiments.",
    }


def _migrate(conn: sqlite3.Connection) -> None:
    # Additive migrations only (safe for a mock DB).
    def _cols() -> set[str]:
        return {r["name"] for r in conn.execute("PRAGMA table_info(loyalty)").fetchall()}

    def _add_column_safe(sql: str) -> None:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as e:
            # Safe in concurrent init/migrate scenarios.
            if "duplicate column name" in str(e).lower():
                return
            raise

    cols = _cols()
    if "cancellation_attempts_12m" not in cols:
        _add_column_safe("ALTER TABLE loyalty ADD COLUMN cancellation_attempts_12m INTEGER NOT NULL DEFAULT 0")
    cols = _cols()
    if "total_spend_usd" not in cols:
        _add_column_safe("ALTER TABLE loyalty ADD COLUMN total_spend_usd REAL NOT NULL DEFAULT 0")

    # Backfill demo-friendly values for known seeded customers if currently all zeros.
    if "cancellation_attempts_12m" in {r["name"] for r in conn.execute("PRAGMA table_info(loyalty)").fetchall()}:
        total = conn.execute("SELECT COALESCE(SUM(cancellation_attempts_12m), 0) AS s FROM loyalty").fetchone()["s"]
        if int(total) == 0:
            backfill = {
                "cust_1001": 1,
                "cust_1002": 0,
                "cust_1003": 1,
                "cust_1004": 2,
            }
            for cid, n in backfill.items():
                conn.execute(
                    "UPDATE loyalty SET cancellation_attempts_12m = ? WHERE customer_id = ?",
                    (n, cid),
                )

    if "total_spend_usd" in {r["name"] for r in conn.execute("PRAGMA table_info(loyalty)").fetchall()}:
        total_spend = conn.execute("SELECT COALESCE(SUM(total_spend_usd), 0) AS s FROM loyalty").fetchone()["s"]
        if float(total_spend) == 0.0:
            backfill_spend = {
                "cust_1001": 45.0,
                "cust_1002": 520.0,
                "cust_1003": 2200.0,
                "cust_1004": 60.0,
            }
            for cid, amt in backfill_spend.items():
                conn.execute(
                    "UPDATE loyalty SET total_spend_usd = ? WHERE customer_id = ?",
                    (amt, cid),
                )

    # Ensure cancellation_events table exists and has at least a little demo data.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cancellation_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          customer_id TEXT NOT NULL,
          event_date TEXT NOT NULL,
          reason TEXT NOT NULL,
          outcome TEXT NOT NULL,
          note TEXT,
          FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        )
        """
    )
    existing_events = conn.execute("SELECT COUNT(*) AS c FROM cancellation_events").fetchone()["c"]
    if int(existing_events) == 0:
        today = date.today()
        demo_events = [
            ("cust_1001", today.replace(day=max(1, min(28, today.day - 12))).isoformat(), "not_using", "saved", None),
            ("cust_1003", today.replace(day=max(1, min(28, today.day - 40))).isoformat(), "too_expensive", "saved", None),
            ("cust_1004", today.replace(day=max(1, min(28, today.day - 18))).isoformat(), "bug_or_quality", "aborted", None),
            ("cust_1004", today.replace(day=max(1, min(28, today.day - 4))).isoformat(), "too_expensive", "saved", None),
        ]
        for ev in demo_events:
            conn.execute(
                "INSERT INTO cancellation_events (customer_id, event_date, reason, outcome, note) VALUES (?, ?, ?, ?, ?)",
                ev,
            )

    # Recompute derived loyalty_score to incorporate any new columns.
    rows = conn.execute("SELECT * FROM loyalty").fetchall()
    for r in rows:
        d = dict(r)
        d["loyalty_score"] = compute_loyalty_score(d)
        conn.execute(
            """
            UPDATE loyalty
            SET loyalty_score = ?, cancellation_attempts_12m = COALESCE(cancellation_attempts_12m, 0)
            WHERE customer_id = ?
            """,
            (d["loyalty_score"], d["customer_id"]),
        )
