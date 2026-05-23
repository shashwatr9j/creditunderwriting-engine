"""
engine.py
═════════════════════════════════════════════════════════════════════════════
Core asynchronous orchestration layer for the Multi-Agent Alternative Credit
Underwriting Engine.

Architecture
------------
        ┌─────────────────────────────────────────────────────────┐
        │                  Ingestion Payload                       │
        │        (bureau_data + device_metadata + sms_logs)        │
        └─────────────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
   ┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │  Agent A       │   │  Agent B         │   │  Agent C         │
   │  Cash-Flow     │   │  Behavioural     │   │  Bureau Cross-   │
   │  Specialist    │   │  Risk Auditor    │   │  Reference       │
   └────────────────┘   └──────────────────┘   └──────────────────┘
            └─────────────────────┼─────────────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │   Lead Underwriter Synthesis  │
                    │   (DTI math + tiering rules)   │
                    └──────────────────────────────┘
                                  ▼
                       Final Decision JSON (Certificate)

The three specialist agents run *concurrently* via ``asyncio.gather``. Each
agent is dual-mode:

  • **LLM mode** — if an Anthropic API key is configured, the agent calls the
    model with a strict JSON-only instruction and parses structured output.
  • **Heuristic mode** — a fully deterministic local analyser that produces the
    same JSON schema. This guarantees the application *always runs* (e.g. in a
    fresh GitHub clone with no API key) and that the UI never crashes on parse.

Both modes return identically-shaped dictionaries, so the synthesis node and
the UI are agnostic to which path produced the data.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# The anthropic SDK is an *optional* runtime dependency. We import it lazily and
# degrade gracefully if it is absent or unconfigured.
try:  # pragma: no cover - import guard
    from anthropic import AsyncAnthropic

    _ANTHROPIC_IMPORTABLE = True
except Exception:  # noqa: BLE001 - any import failure means LLM mode is off
    AsyncAnthropic = None  # type: ignore
    _ANTHROPIC_IMPORTABLE = False

# Optional .env loading so a local key is picked up automatically.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.environ.get("UNDERWRITING_MODEL", "claude-sonnet-4-6")
BASE_INTEREST_RATE = float(os.environ.get("BASE_INTEREST_RATE", "10.5"))  # annual %

# Tiering thresholds (kept as named constants — no magic numbers in logic).
TIER1_MAX_DTI = 0.35
TIER2_MAX_DTI = 0.50
TIER1_MIN_CIBIL = 750
TIER2_MIN_CIBIL = 650

TIER1_MAX_LOAN = 500_000
TIER2_MAX_LOAN = 150_000

TIER1_RATE_SPREAD = 1.5
TIER2_RATE_SPREAD = 3.5


def llm_available() -> bool:
    """True only if the SDK is importable *and* an API key is present."""
    return _ANTHROPIC_IMPORTABLE and bool(os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight SMS parsing helpers (shared by heuristic agents)
# ─────────────────────────────────────────────────────────────────────────────
_AMOUNT_RE = re.compile(r"(?:INR|Rs\.?)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)

_BNPL_KEYWORDS = ["slice", "uni", "postpe", "lazypay", "simpl", "zestmoney"]
_GAMBLING_KEYWORDS = ["bet365", "dream11", "mpl", "parimatch", "rummycircle", "betway"]
_SUBSCRIPTION_KEYWORDS = [
    "netflix", "spotify", "prime", "hotstar", "cult.fit", "youtube premium", "auto-pay",
]
_CREDIT_KEYWORDS = ["credited", "credit of", "salary"]
_FAILED_KEYWORDS = ["failed", "insufficient balance", "bounce", "declined"]


def _extract_amount(text: str) -> Optional[float]:
    """Pull the first INR/Rs amount out of an SMS string, if present."""
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _is_late_night(text: str) -> bool:
    """Detect a 00:00–04:59 or 23:xx timestamp inside the SMS string."""
    m = re.search(r"\b([0-2]?\d):[0-5]\d\b", text)
    if not m:
        return False
    try:
        hour = int(m.group(1))
    except ValueError:
        return False
    return hour <= 4 or hour == 23


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AgentResult:
    """Uniform wrapper around any agent's structured output."""

    agent_id: str
    agent_name: str
    mode: str                       # "llm" or "heuristic"
    output: Dict[str, Any]
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "mode": self.mode,
            "output": self.output,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
        }


@dataclass
class UnderwritingResult:
    """Complete result bundle returned to the UI."""

    profile_id: str
    agent_results: List[AgentResult] = field(default_factory=list)
    decision: Dict[str, Any] = field(default_factory=dict)
    engine_mode: str = "heuristic"
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "engine_mode": self.engine_mode,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "agents": [a.to_dict() for a in self.agent_results],
            "decision": self.decision,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction for LLM mode
# ─────────────────────────────────────────────────────────────────────────────
def _agent_a_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are Agent A, a Cash-Flow Specialist in a credit underwriting "
        "pipeline. Analyse the raw transaction SMS logs and bureau income.\n\n"
        f"BUREAU_DATA:\n{json.dumps(payload['bureau_data'], indent=2)}\n\n"
        f"SMS_LOGS:\n{json.dumps(payload['sms_logs'], indent=2)}\n\n"
        "Tasks:\n"
        "1. Estimate verified gross monthly income from credit/salary SMS.\n"
        "2. Detect income regularity (regular vs irregular/gig).\n"
        "3. Sum recurring subscription liabilities.\n"
        "4. Estimate average monthly discretionary spend.\n"
        "5. Compute monthly disposable margin = income - (subscriptions + "
        "discretionary spend + bureau EMI).\n\n"
        "Respond with ONLY a JSON object using these exact keys: "
        "verified_monthly_income (number), income_regularity (string: "
        "'regular' or 'irregular'), recurring_subscription_total (number), "
        "estimated_monthly_discretionary (number), monthly_disposable_margin "
        "(number), notes (array of short strings)."
    )


def _agent_b_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are Agent B, a Behavioural Risk Auditor in a credit underwriting "
        "pipeline. Analyse SMS logs and device metadata for warning flags.\n\n"
        f"DEVICE_METADATA:\n{json.dumps(payload['device_metadata'], indent=2)}\n\n"
        f"SMS_LOGS:\n{json.dumps(payload['sms_logs'], indent=2)}\n\n"
        "Look for: (a) high-frequency late-night gambling merchant activity, "
        "(b) reliance on multiple distinct BNPL apps (slice/uni/postpe/etc), "
        "(c) cash-crunch signals like failed auto-debits or low balance.\n\n"
        "Respond with ONLY a JSON object using these exact keys: "
        "gambling_txn_count (integer), gambling_total_amount (number), "
        "distinct_bnpl_apps (integer), bnpl_outstanding_total (number), "
        "cash_crunch_signals (integer), behavioural_risk_score (number 0-100, "
        "higher = riskier), severe_flag (boolean), flags (array of short "
        "strings)."
    )


def _agent_c_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are Agent C, a Bureau Cross-Reference analyst in a credit "
        "underwriting pipeline. Correlate informal data with bureau metrics.\n\n"
        f"BUREAU_DATA:\n{json.dumps(payload['bureau_data'], indent=2)}\n\n"
        f"SMS_LOGS:\n{json.dumps(payload['sms_logs'], indent=2)}\n\n"
        "Detect credit hunger: high recent hard inquiries with few/zero new "
        "loan originations visible in cash flow. Flag mismatch between bureau-"
        "reported income and SMS-observed income.\n\n"
        "Respond with ONLY a JSON object using these exact keys: "
        "credit_hunger_flag (boolean), inquiry_to_origination_ratio (number), "
        "income_mismatch_pct (number), bureau_consistency_score (number 0-100, "
        "higher = more consistent), notes (array of short strings)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM call wrapper
# ─────────────────────────────────────────────────────────────────────────────
async def _call_llm_json(prompt: str, model: str) -> Dict[str, Any]:
    """
    Call the Anthropic API and return parsed JSON. Raises on any failure so the
    caller can fall back to heuristics. The client is closed explicitly to
    avoid dangling-connection warnings on the event loop.
    """
    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=1024,
            system=(
                "You are a precise financial analysis agent. You ALWAYS respond "
                "with a single valid JSON object and nothing else — no markdown "
                "fences, no preamble, no commentary."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate all text blocks defensively.
        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
        return _safe_json_loads(text)
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Robustly parse a JSON object from model text. Strips markdown fences and
    extracts the outermost {...} span if extra prose sneaks in.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic (deterministic) agent implementations
# ─────────────────────────────────────────────────────────────────────────────
def _heuristic_agent_a(payload: Dict[str, Any]) -> Dict[str, Any]:
    sms = payload["sms_logs"]
    bureau = payload["bureau_data"]

    credit_amounts: List[float] = []
    subscription_total = 0.0
    discretionary_total = 0.0
    notes: List[str] = []

    for line in sms:
        low = line.lower()
        amt = _extract_amount(line)
        if amt is None:
            continue
        if any(k in low for k in _CREDIT_KEYWORDS):
            credit_amounts.append(amt)
        elif any(k in low for k in _SUBSCRIPTION_KEYWORDS):
            subscription_total += amt
        elif "debited" in low and not any(g in low for g in _GAMBLING_KEYWORDS):
            discretionary_total += amt

    verified_income = sum(credit_amounts) if credit_amounts else float(
        bureau.get("reported_gross_monthly_income", 0.0)
    )

    # Regularity: a single large credit ≈ salary; many uneven credits ≈ gig.
    if len(credit_amounts) <= 1:
        regularity = "regular"
    else:
        mean_credit = sum(credit_amounts) / len(credit_amounts)
        spread = max(credit_amounts) - min(credit_amounts)
        # High relative spread across multiple credits => irregular/gig income.
        regularity = "irregular" if (mean_credit > 0 and spread > 0.5 * mean_credit) else "regular"
        notes.append(f"{len(credit_amounts)} income credits observed ({regularity})")

    bureau_emi = float(bureau.get("total_monthly_emi", 0.0))
    disposable = verified_income - (subscription_total + discretionary_total + bureau_emi)

    notes.append(f"Subscriptions detected: INR {subscription_total:,.0f}")
    if disposable < 0:
        notes.append("Negative disposable margin — outflows exceed inflows")

    return {
        "verified_monthly_income": round(verified_income, 2),
        "income_regularity": regularity,
        "recurring_subscription_total": round(subscription_total, 2),
        "estimated_monthly_discretionary": round(discretionary_total, 2),
        "monthly_disposable_margin": round(disposable, 2),
        "notes": notes,
    }


def _heuristic_agent_b(payload: Dict[str, Any]) -> Dict[str, Any]:
    sms = payload["sms_logs"]
    device = payload["device_metadata"]

    gambling_count = 0
    gambling_total = 0.0
    bnpl_apps: set[str] = set()
    bnpl_total = 0.0
    cash_crunch = 0
    flags: List[str] = []

    for line in sms:
        low = line.lower()
        amt = _extract_amount(line) or 0.0
        if any(g in low for g in _GAMBLING_KEYWORDS):
            gambling_count += 1
            gambling_total += amt
        for app in _BNPL_KEYWORDS:
            if app in low:
                bnpl_apps.add(app)
                if "bill" in low or "due" in low:
                    bnpl_total += amt
        if any(f in low for f in _FAILED_KEYWORDS):
            cash_crunch += 1

    late_night_hours = int(device.get("late_night_active_hours", 0))
    wallet_apps = int(device.get("active_wallet_apps", 0))

    # Composite behavioural risk score (0-100, higher = riskier).
    score = 0.0
    score += min(gambling_count * 8, 40)
    score += min(len(bnpl_apps) * 7, 28)
    score += min(cash_crunch * 6, 18)
    score += min(max(late_night_hours - 2, 0) * 2, 8)
    score += min(max(wallet_apps - 4, 0) * 1.5, 6)
    if device.get("is_rooted_device"):
        score += 5
    score = min(round(score, 1), 100.0)

    if gambling_count >= 3:
        flags.append(f"High-frequency gambling: {gambling_count} transactions")
    elif gambling_count > 0:
        flags.append(f"Gambling activity detected: {gambling_count} transactions")
    if len(bnpl_apps) >= 3:
        flags.append(f"BNPL stacking across {len(bnpl_apps)} apps")
    elif bnpl_apps:
        flags.append(f"BNPL usage: {len(bnpl_apps)} app(s)")
    if cash_crunch:
        flags.append(f"Cash-crunch signals: {cash_crunch} failed/low-balance event(s)")
    if late_night_hours >= 4:
        flags.append(f"Elevated late-night activity ({late_night_hours}h)")
    if device.get("is_rooted_device"):
        flags.append("Rooted device detected")
    if not flags:
        flags.append("No material behavioural flags")

    # Severe flag drives Tier-3 rejection downstream. Reserve "severe" for
    # genuinely high-risk patterns so moderate BNPL usage can land in Tier-2.
    severe = gambling_count >= 3 or len(bnpl_apps) >= 5 or score >= 60

    return {
        "gambling_txn_count": gambling_count,
        "gambling_total_amount": round(gambling_total, 2),
        "distinct_bnpl_apps": len(bnpl_apps),
        "bnpl_outstanding_total": round(bnpl_total, 2),
        "cash_crunch_signals": cash_crunch,
        "behavioural_risk_score": score,
        "severe_flag": severe,
        "flags": flags,
    }


def _heuristic_agent_c(payload: Dict[str, Any]) -> Dict[str, Any]:
    sms = payload["sms_logs"]
    bureau = payload["bureau_data"]

    hard_inquiries = int(bureau.get("hard_inquiries_30d", 0))
    active_loans = int(bureau.get("active_loans", 0))
    reported_income = float(bureau.get("reported_gross_monthly_income", 0.0))

    # Observed income from SMS credits.
    observed_income = 0.0
    for line in sms:
        low = line.lower()
        if any(k in low for k in _CREDIT_KEYWORDS):
            observed_income += _extract_amount(line) or 0.0

    # Inquiry-to-origination ratio: many pulls, few loans => credit hunger.
    ratio = hard_inquiries / (active_loans + 1)
    credit_hunger = hard_inquiries >= 4 and ratio >= 2.0

    if reported_income > 0 and observed_income > 0:
        income_mismatch = abs(reported_income - observed_income) / reported_income * 100
    else:
        income_mismatch = 0.0

    consistency = 100.0
    consistency -= min(hard_inquiries * 4, 30)
    consistency -= min(income_mismatch * 0.5, 30)
    if credit_hunger:
        consistency -= 15
    consistency = max(round(consistency, 1), 0.0)

    notes: List[str] = []
    if credit_hunger:
        notes.append(
            f"Credit hunger: {hard_inquiries} inquiries vs {active_loans} active loans"
        )
    if income_mismatch > 25:
        notes.append(f"Income mismatch {income_mismatch:.0f}% (bureau vs observed)")
    if not notes:
        notes.append("Bureau data broadly consistent with cash flow")

    return {
        "credit_hunger_flag": credit_hunger,
        "inquiry_to_origination_ratio": round(ratio, 2),
        "income_mismatch_pct": round(income_mismatch, 1),
        "bureau_consistency_score": consistency,
        "notes": notes,
    }


# Map each agent to (id, name, prompt_builder, heuristic_fn).
_AGENT_REGISTRY = [
    ("agent_a", "Cash-Flow Specialist", _agent_a_prompt, _heuristic_agent_a),
    ("agent_b", "Behavioural Risk Auditor", _agent_b_prompt, _heuristic_agent_b),
    ("agent_c", "Bureau Cross-Reference", _agent_c_prompt, _heuristic_agent_c),
]


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent runner (dual-mode)
# ─────────────────────────────────────────────────────────────────────────────
async def _run_agent(
    agent_id: str,
    agent_name: str,
    prompt_builder: Callable[[Dict[str, Any]], str],
    heuristic_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
    use_llm: bool,
    model: str,
) -> AgentResult:
    """Execute one agent, preferring LLM mode but falling back to heuristics."""
    start = time.perf_counter()

    if use_llm:
        try:
            output = await _call_llm_json(prompt_builder(payload), model)
            # Merge with heuristic defaults to guarantee required keys exist.
            merged = heuristic_fn(payload)
            merged.update({k: v for k, v in output.items() if v is not None})
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(agent_id, agent_name, "llm", merged, duration)
        except Exception as exc:  # noqa: BLE001 - fall back on any LLM failure
            # Simulate a small amount of work for UI realism, then heuristic.
            await asyncio.sleep(0.05)
            output = heuristic_fn(payload)
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                agent_id, agent_name, "heuristic", output, duration,
                error=f"LLM fallback: {type(exc).__name__}: {exc}",
            )

    # Pure heuristic mode — add a tiny async yield so agents truly interleave.
    await asyncio.sleep(0.05)
    output = heuristic_fn(payload)
    duration = (time.perf_counter() - start) * 1000
    return AgentResult(agent_id, agent_name, "heuristic", output, duration)


# ─────────────────────────────────────────────────────────────────────────────
# Lead Underwriter synthesis node
# ─────────────────────────────────────────────────────────────────────────────
def synthesize_decision(
    payload: Dict[str, Any],
    agent_a: Dict[str, Any],
    agent_b: Dict[str, Any],
    agent_c: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Combine the three agent outputs into a final, rule-based credit decision.

    DTI = (bureau EMIs + hidden BNPL liabilities) / gross monthly income
    """
    bureau = payload["bureau_data"]
    cibil = int(bureau.get("cibil_score", 0))
    bureau_emi = float(bureau.get("total_monthly_emi", 0.0))

    gross_income = float(agent_a.get("verified_monthly_income", 0.0)) or float(
        bureau.get("reported_gross_monthly_income", 0.0)
    )

    # Hidden liabilities = subscriptions + an amortised slice of BNPL outstanding.
    hidden_subscriptions = float(agent_a.get("recurring_subscription_total", 0.0))
    bnpl_outstanding = float(agent_b.get("bnpl_outstanding_total", 0.0))
    # Treat BNPL bills as near-term monthly obligations (already monthly here).
    hidden_liabilities = hidden_subscriptions + bnpl_outstanding

    total_monthly_obligations = bureau_emi + hidden_liabilities

    if gross_income > 0:
        dti = total_monthly_obligations / gross_income
    else:
        dti = 1.0  # no verifiable income => maximal risk

    dti_pct = round(dti * 100, 1)

    severe_behavioural = bool(agent_b.get("severe_flag", False))
    behavioural_score = float(agent_b.get("behavioural_risk_score", 0.0))
    credit_hunger = bool(agent_c.get("credit_hunger_flag", False))

    reasons: List[str] = []

    # ── Tier determination ────────────────────────────────────────────────────
    if dti > TIER2_MAX_DTI or severe_behavioural or cibil < TIER2_MIN_CIBIL:
        tier = "Tier-3"
        tier_label = "High Risk"
        status = "Rejected"
        max_loan = 0
        interest_rate = None
        if dti > TIER2_MAX_DTI:
            reasons.append(f"DTI {dti_pct}% exceeds 50% ceiling")
        if severe_behavioural:
            reasons.append("Severe behavioural risk flag raised by Agent B")
        if cibil < TIER2_MIN_CIBIL:
            reasons.append(f"CIBIL {cibil} below minimum threshold of 650")
        if credit_hunger:
            reasons.append("Credit-hunger pattern detected by Agent C")

    elif (
        dti < TIER1_MAX_DTI
        and cibil > TIER1_MIN_CIBIL
        and behavioural_score < 25
        and not credit_hunger
    ):
        tier = "Tier-1"
        tier_label = "Low Risk"
        status = "Approved"
        max_loan = TIER1_MAX_LOAN
        interest_rate = round(BASE_INTEREST_RATE + TIER1_RATE_SPREAD, 2)
        reasons.append(f"DTI {dti_pct}% within prime band (<35%)")
        reasons.append(f"CIBIL {cibil} above prime threshold (>750)")
        reasons.append("Clean behavioural profile")

    else:
        tier = "Tier-2"
        tier_label = "Moderate Risk"
        status = "Approved"
        max_loan = TIER2_MAX_LOAN
        interest_rate = round(BASE_INTEREST_RATE + TIER2_RATE_SPREAD, 2)
        reasons.append(f"DTI {dti_pct}% within moderate band (35%-50%)")
        reasons.append(f"CIBIL {cibil} in moderate range")
        if behavioural_score >= 25:
            reasons.append(
                f"Minor behavioural flags (risk score {behavioural_score:.0f}/100)"
            )

    return {
        "profile_id": payload.get("meta", {}).get("profile_id", "UNKNOWN"),
        "risk_tier": tier,
        "risk_tier_label": tier_label,
        "status": status,
        "max_loan_amount": max_loan,
        "interest_rate_pct": interest_rate,
        "base_rate_pct": BASE_INTEREST_RATE,
        "dti_pct": dti_pct,
        "financials": {
            "gross_monthly_income": round(gross_income, 2),
            "bureau_monthly_emi": round(bureau_emi, 2),
            "hidden_liabilities": round(hidden_liabilities, 2),
            "total_monthly_obligations": round(total_monthly_obligations, 2),
            "cibil_score": cibil,
            "behavioural_risk_score": behavioural_score,
        },
        "reasons": reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────
async def run_underwriting(
    payload: Dict[str, Any],
    use_llm: Optional[bool] = None,
    model: str = DEFAULT_MODEL,
) -> UnderwritingResult:
    """
    Run the full multi-agent underwriting pipeline.

    Parameters
    ----------
    payload : dict
        A borrower profile (see data_generator.generate_profile).
    use_llm : bool | None
        Force LLM mode on/off. If None, auto-detect via ``llm_available()``.
    model : str
        Anthropic model identifier for LLM mode.
    """
    _validate_payload(payload)

    if use_llm is None:
        use_llm = llm_available()

    pipeline_start = time.perf_counter()

    # ── Stage 1: run the three specialists concurrently ──────────────────────
    tasks = [
        _run_agent(aid, name, pb, hf, payload, use_llm, model)
        for (aid, name, pb, hf) in _AGENT_REGISTRY
    ]
    agent_results: List[AgentResult] = await asyncio.gather(*tasks)

    by_id = {r.agent_id: r.output for r in agent_results}

    # ── Stage 2: lead underwriter synthesis ──────────────────────────────────
    decision = synthesize_decision(
        payload, by_id["agent_a"], by_id["agent_b"], by_id["agent_c"]
    )

    total_ms = (time.perf_counter() - pipeline_start) * 1000

    # Engine mode is "llm" only if at least one agent actually used the LLM.
    any_llm = any(r.mode == "llm" for r in agent_results)
    engine_mode = "llm" if any_llm else "heuristic"

    return UnderwritingResult(
        profile_id=payload.get("meta", {}).get("profile_id", "UNKNOWN"),
        agent_results=agent_results,
        decision=decision,
        engine_mode=engine_mode,
        total_duration_ms=total_ms,
    )


def run_underwriting_sync(
    payload: Dict[str, Any],
    use_llm: Optional[bool] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Synchronous convenience wrapper for Streamlit (which is not async-native).
    Returns a plain dict ready for the UI.
    """
    result = asyncio.run(run_underwriting(payload, use_llm=use_llm, model=model))
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def _validate_payload(payload: Dict[str, Any]) -> None:
    """Raise a clear error if the ingestion payload is malformed."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    for key in ("bureau_data", "device_metadata", "sms_logs"):
        if key not in payload:
            raise ValueError(f"payload missing required section: '{key}'")
    if not isinstance(payload["sms_logs"], list):
        raise ValueError("payload['sms_logs'] must be a list of strings")


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_generator import generate_all_profiles

    profiles = generate_all_profiles()
    for name, prof in profiles.items():
        res = run_underwriting_sync(prof, use_llm=False)
        d = res["decision"]
        print(
            f"{name:>10} | {d['risk_tier']:<7} {d['status']:<9} "
            f"DTI={d['dti_pct']:>5}% CIBIL={d['financials']['cibil_score']} "
            f"limit=INR {d['max_loan_amount']:,}"
        )
