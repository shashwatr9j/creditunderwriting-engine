# ◆ Multi-Agent Alternative Credit Underwriting Engine

> An asynchronous, multi-agent decisioning system that underwrites digital
> consumer loans from **messy alternative data** — raw transaction SMS,
> device telemetry, and a thin bureau snapshot — without waiting on a manual
> credit-bureau pull.

Built for the **instant digital-lending** use case: 15-minute financing lines
where an NBFC must form a defensible credit view in seconds from noisy,
real-world signals.

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=shashwatr9j/underwriting-engine&branch=master&mainModule=app.py)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-shashwatr9j%2Funderwriting--engine-181717?logo=github)](https://github.com/shashwatr9j/underwriting-engine)

---

## Why this exists

Traditional underwriting leans almost entirely on a clean CIBIL pull and a
salary slip. That model breaks for two large, fast-growing borrower segments:

- **Thin-file borrowers** — gig workers, first-jobbers, and BNPL-native users
  with little formal credit history but a rich digital transaction footprint.
- **Speed-sensitive originations** — checkout-finance and instant-loan flows
  where a multi-day manual review is commercially impossible.

This engine reads the *alternative* footprint (transaction SMS, recurring
liabilities, behavioural signals) in parallel, reconciles it against whatever
bureau data exists, and produces a structured, auditable credit decision.

---

## System Architecture

The engine uses a **fan-out / fan-in** orchestration pattern. Three
specialist agents analyse the same ingestion payload concurrently, then a Lead
Underwriter node synthesises their structured outputs into a final decision.

```
                ┌──────────────────────────────────────────────┐
                │             Ingestion Payload                 │
                │   bureau_data · device_metadata · sms_logs    │
                └──────────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
     │   Agent A       │   │   Agent B        │   │   Agent C        │
     │   Cash-Flow     │   │   Behavioural    │   │   Bureau Cross-  │
     │   Specialist    │   │   Risk Auditor   │   │   Reference      │
     │                 │   │                  │   │                  │
     │ • verified inc. │   │ • gambling flags │   │ • credit hunger  │
     │ • subscriptions │   │ • BNPL stacking  │   │ • inquiry ratio  │
     │ • disposable    │   │ • cash-crunch    │   │ • income mismatch│
     └─────────────────┘   └──────────────────┘   └──────────────────┘
              └──────────────────────┼──────────────────────┘
                                     ▼
                       ┌────────────────────────────┐
                       │   Lead Underwriter Node      │
                       │   • DTI computation          │
                       │   • Tiering rules            │
                       │   • Sanction limit & pricing │
                       └────────────────────────────┘
                                     ▼
                        Credit Decision Certificate (JSON)
```

The three specialists run via `asyncio.gather`, so wall-clock latency is
bounded by the **slowest single agent**, not the sum of all three.

---

## The Agents

| Agent | Role | Consumes | Produces |
|-------|------|----------|----------|
| **A — Cash-Flow Specialist** | Reconstructs income & liabilities from raw SMS | `sms_logs`, `bureau_data` | verified income, income regularity, subscription total, discretionary spend, **monthly disposable margin** |
| **B — Behavioural Risk Auditor** | Surfaces conduct red flags | `sms_logs`, `device_metadata` | gambling txn count/value, distinct BNPL apps, cash-crunch signals, **behavioural risk score (0–100)**, severe-flag boolean |
| **C — Bureau Cross-Reference** | Reconciles informal vs formal data | `sms_logs`, `bureau_data` | credit-hunger flag, inquiry-to-origination ratio, income mismatch %, bureau-consistency score |

### Lead Underwriter — decision math

The synthesis node computes Debt-to-Income from **both** formal and
SMS-discovered ("hidden") liabilities:

```
DTI = (Bureau Monthly EMIs + Hidden Liabilities) / Gross Monthly Income

where Hidden Liabilities = recurring subscriptions + BNPL outstanding
```

It then assigns a risk tier:

| Tier | Conditions | Max Loan | Interest Rate |
|------|-----------|----------|---------------|
| **Tier-1 · Low Risk** | DTI < 35% **and** CIBIL > 750 **and** clean behavioural profile | ₹5,00,000 | Base + 1.5% |
| **Tier-2 · Moderate** | DTI 35–50%, minor flags, CIBIL 650–750 | ₹1,50,000 | Base + 3.5% |
| **Tier-3 · High Risk** | DTI > 50% **or** severe behavioural flag **or** CIBIL < 650 | — | **Rejected** |

The base rate defaults to **10.5%** and is configurable via the
`BASE_INTEREST_RATE` environment variable.

---

## Dual-Mode Reasoning (LLM + Heuristic)

Every agent is **dual-mode** and returns an identical JSON schema either way:

- **LLM mode** — if an `ANTHROPIC_API_KEY` is configured, each agent calls the
  Claude API with a strict JSON-only instruction and parses structured output.
- **Heuristic mode** — a fully deterministic local analyser (regex + rules).

If an LLM call fails for any reason (no key, network, rate limit, malformed
output), the agent **silently and gracefully falls back** to its heuristic
implementation. The consequence: **the application always runs and the UI
never crashes on a parse error**, even on a fresh clone with no credentials.

This is also why the demo is fully reproducible offline.

---

## Project Structure

```
underwriting_engine/
├── requirements.txt      # Locked dependencies
├── data_generator.py     # Synthetic borrower-profile generator (3 archetypes)
├── engine.py             # Async orchestration + agents + synthesis node
├── app.py                # Streamlit dual-panel cockpit UI
└── README.md             # You are here
```

Each module is import-safe and independently runnable:

```bash
python data_generator.py --archetype high_risk --seed 7   # print a profile
python engine.py                                           # CLI smoke test
streamlit run app.py                                       # full UI
```

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone <your-repo-url>
cd underwriting_engine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Enable LLM agents

The app runs fully without this step. To enable Claude-powered agents, create
a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
# Optional overrides:
# UNDERWRITING_MODEL=claude-sonnet-4-6
# BASE_INTEREST_RATE=10.5
```

### 4. Launch

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (default `http://localhost:8501`).

---

## Using the Dashboard

The cockpit is split into two panels:

**Left — Ingestion Console**
- Choose **Mock profile** and pick one of three archetypes, or
- Choose **Paste raw SMS** and drop in a free-text block of transaction alerts
  (one per line). Bureau fields are unknown for a cold paste, so the decision
  leans on observed cash flow.
- Hit **▶ Execute Underwriting Engine**.

**Right — Agent Telemetry**
- Live, expandable panels for Agents A / B / C showing each agent's mode
  (LLM/heuristic), latency, structured metrics, flags, and raw JSON.
- A **Credit Decision Certificate** card with the risk tier, sanction limit,
  interest rate, DTI, the full DTI computation table, and the itemised
  approval/rejection rationale.

---

## Synthetic Data Archetypes

`data_generator.py` produces three deliberately distinct, chaotic profiles:

1. **Prime Borrower** — high, regular salary; clean SMS feed; CIBIL 760–830.
   → consistently underwrites to **Tier-1**.
2. **Thin-File Borrower** — irregular gig income; multiple BNPL bills; modest
   CIBIL. → underwrites to **Tier-2 / Tier-3** depending on stacking depth.
3. **High-Risk Borrower** — volatile late-night gambling, credit hunger,
   elevated EMIs. → consistently **Tier-3 / Rejected**.

SMS strings are intentionally inconsistent in format (varied timestamps,
`INR` vs `Rs`, mixed casing) to mirror the noise of a real telecom feed.

---

## Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | _unset_ | Enables LLM-mode agents. Unset ⇒ heuristic mode. |
| `UNDERWRITING_MODEL` | `claude-sonnet-4-6` | Model ID for LLM mode. |
| `BASE_INTEREST_RATE` | `10.5` | Annual base rate (%) before tier spread. |

---

## Design Notes & Limitations

- **Not financial advice.** This is a reference architecture and demonstration,
  not a regulated production underwriting system. Tiering thresholds, rates,
  and limits are illustrative and must be calibrated and validated against real
  portfolio performance, fair-lending requirements, and applicable regulation
  before any production use.
- **Synthetic data only.** No real borrower data is used or required.
- **Heuristics are intentionally transparent.** The local analysers favour
  explainability over sophistication; they are a deterministic baseline, not a
  trained risk model.
- **Determinism.** With a fixed `--seed`, profiles and (heuristic-mode)
  decisions are fully reproducible.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive dashboard
- **asyncio** — parallel agent orchestration
- **Anthropic SDK** — optional LLM reasoning
- **pandas** — tabular rendering

---

## Reference Data

The repo ships three ready-to-use sample files so you can try every workflow
without writing a single line of code.

### `sample_profiles.json`

A single JSON file containing all three archetype profiles in the exact schema
the engine expects. Load any profile directly in Python:

```python
import json
from engine import run_underwriting_sync

with open("sample_profiles.json") as f:
    profiles = json.load(f)

# pick one: "prime" | "thin_file" | "high_risk"
result = run_underwriting_sync(profiles["high_risk"], use_llm=False)
print(result["decision"]["status"])       # → Rejected
print(result["decision"]["risk_tier"])    # → Tier-3
```

### `sample_sms_*.txt` — paste-mode samples

Three plain-text SMS dumps (one per archetype) ready to paste into the
**Paste raw SMS** panel in the UI:

| File | Archetype | Expected outcome |
|------|-----------|-----------------|
| `sample_sms_prime.txt` | High-earner, salaried | Tier-1 · Approved |
| `sample_sms_thin_file.txt` | Gig income, BNPL-stacked | Tier-2 · Approved |
| `sample_sms_high_risk.txt` | Gambling + credit hunger | Tier-3 · Rejected |

Copy any file's contents, select **Paste raw SMS** in the UI, and hit
**▶ Execute Underwriting Engine**.

### CLI smoke test (all archetypes, deterministic)

```bash
python engine.py
#  prime | Tier-1  Approved  DTI=  2.3% CIBIL=796 limit=INR 500,000
#  thin_file | Tier-2  Approved  DTI= 35.8% CIBIL=706 limit=INR 150,000
#  high_risk | Tier-3  Rejected  DTI= 50.5% CIBIL=613 limit=INR 0
```

---

## License

Provided as-is for demonstration and educational purposes. Add your preferred
license file before distribution.
