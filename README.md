# Subscription “Save Offer” Agent (Commerce PM Prototype)

Agentic prototype for a **subscription cancellation save flow** (Option 2: build an agent for a persona).

**Persona:** Retention / Support agent handling “I want to cancel” requests.  
**Job-to-be-done:** Recommend the *smallest effective* save offer that balances retention, margin, and policy constraints, and explain why.

## What this demonstrates
- **Agentic reasoning with tools:** the model gathers context via tool calls (customer, subscription, usage, eligible offers, impact simulation).
- **Decision-making:** chooses an offer with explicit tradeoffs + alternatives.
- **Working prototype:** simple web UI + JSON API; mock data is clearly labeled.

## Run locally
1) Create a venv and install deps:
   - `cd subscription-save-agent`
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`

2) Configure environment:
   - `cp .env.example .env`
   - Set `OPENAI_API_KEY=...` in `.env`

3) Start the server:
   - `uvicorn app.main:app --reload --port 8000`

4) Open:
   - `http://localhost:8000`

## Deployment notes
This app is compatible with common Python hosts (Render/Railway/Fly/Heroku-like).

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Env vars: `OPENAI_API_KEY`, optionally `OPENAI_MODEL`

## Replit hosting (recommended for quick URL)
Yes—import the code into Replit.

1) In Replit, create a new **Python** repl (or “Import from GitHub”).
2) Upload/import the `subscription-save-agent/` folder as the repl contents (so `requirements.txt` is at the project root in Replit).
3) In Replit **Secrets**, add:
   - `OPENAI_API_KEY` = your key
   - (optional) `OPENAI_MODEL` = `o4-mini`
4) Click **Run**. Replit uses `.replit` to start `uvicorn`.
5) Use the public Replit URL and verify `/` loads.

## Mock vs real
- **Real:** OpenAI model call (when `OPENAI_API_KEY` is set).
- **Mocked:** customer/subscription/usage data, offer catalog, and impact simulation (clearly shown in the UI “Audit log”).
