"""
app.py
═════════════════════════════════════════════════════════════════════════════
Streamlit frontend for the Multi-Agent Alternative Credit Underwriting Engine.

A dual-panel underwriting cockpit:

  • LEFT  — profile selection (pre-generated archetypes) or a free-text paste
            of raw transaction SMS, plus the "Execute Underwriting Engine"
            trigger and engine-mode controls.
  • RIGHT — live, expandable reasoning panels for Agents A / B / C running in
            parallel, followed by a high-end "Credit Decision Certificate".

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data_generator import (
    ARCHETYPE_LABELS,
    ARCHETYPES,
    generate_all_profiles,
    generate_profile,
)
from engine import BASE_INTEREST_RATE, llm_available, run_underwriting_sync

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration & theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alt-Credit Underwriting Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A restrained, finance-grade visual identity: deep ink navy, parchment surface,
# and a single signal-gold accent. Tier colours are semantic (green/amber/red).
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0f1c2e;
    --ink-soft:   #2a3a4f;
    --surface:    #f7f4ec;
    --surface-2:  #ffffff;
    --line:       #d9d2c4;
    --gold:       #b8893b;
    --green:      #1f7a55;
    --amber:      #b5791f;
    --red:        #a32f2f;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main .block-container {
    padding-top: 2.2rem;
    max-width: 1400px;
}

h1, h2, h3, .uw-display {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.01em;
}

.uw-header {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.6rem;
    margin-bottom: 1.4rem;
}
.uw-header .uw-title {
    font-family: 'Fraunces', serif;
    font-size: 2.05rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.1;
}
.uw-header .uw-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    color: var(--ink-soft);
    text-transform: uppercase;
    margin-top: 0.3rem;
}

.uw-panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* Decision certificate card */
.uw-cert {
    border: 1.5px solid var(--ink);
    border-radius: 4px;
    background: var(--surface-2);
    padding: 1.6rem 1.8rem;
    margin-top: 0.4rem;
    box-shadow: 0 18px 40px -28px rgba(15,28,46,0.55);
    position: relative;
}
.uw-cert::before {
    content: "";
    position: absolute; inset: 6px;
    border: 1px solid var(--line);
    border-radius: 2px;
    pointer-events: none;
}
.uw-cert-head {
    display: flex; justify-content: space-between; align-items: flex-start;
    border-bottom: 1px dashed var(--line);
    padding-bottom: 0.9rem; margin-bottom: 1.1rem;
}
.uw-cert-kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--ink-soft);
}
.uw-cert-title {
    font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600;
    color: var(--ink); margin-top: 0.15rem;
}
.uw-badge {
    display: inline-block; padding: 0.35rem 0.85rem; border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 500;
    font-size: 0.82rem; letter-spacing: 0.05em; color: #fff;
}
.uw-badge.t1 { background: var(--green); }
.uw-badge.t2 { background: var(--amber); }
.uw-badge.t3 { background: var(--red); }

.uw-metric-row { display: flex; gap: 1.4rem; flex-wrap: wrap; margin: 0.4rem 0 1.0rem; }
.uw-metric { flex: 1; min-width: 130px; }
.uw-metric .lbl {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-soft);
}
.uw-metric .val {
    font-family: 'Fraunces', serif; font-size: 1.65rem; font-weight: 600;
    color: var(--ink); line-height: 1.2;
}
.uw-metric .val.green { color: var(--green); }
.uw-metric .val.amber { color: var(--amber); }
.uw-metric .val.red   { color: var(--red); }

.uw-mode-pill {
    display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.66rem;
    letter-spacing:0.08em; padding:0.2rem 0.6rem; border-radius:999px;
    border:1px solid var(--line); color:var(--ink-soft); background:var(--surface);
}
.uw-foot {
    font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:var(--ink-soft);
    border-top:1px dashed var(--line); padding-top:0.7rem; margin-top:1.0rem;
    display:flex; justify-content:space-between;
}
.stExpander {
    border:1px solid var(--line) !important; border-radius:4px !important;
    background: var(--surface-2) !important;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "demo_profiles" not in st.session_state:
    st.session_state.demo_profiles = generate_all_profiles(seed=42)
if "result" not in st.session_state:
    st.session_state.result = None
if "active_payload" not in st.session_state:
    st.session_state.active_payload = None


# ─────────────────────────────────────────────────────────────────────────────
# Raw SMS dump → payload parser (for the manual paste workflow)
# ─────────────────────────────────────────────────────────────────────────────
def build_payload_from_raw_sms(raw_text: str) -> Dict[str, Any]:
    """
    Convert a free-text paste of SMS lines into a minimal valid payload.

    Bureau and device sections are unknown for a raw paste, so we seed neutral
    defaults and let Agent A infer income from the SMS credits themselves. This
    keeps the engine contract satisfied without fabricating a bureau score.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # Best-effort late-night detection from timestamps in the paste.
    late_night = 0
    for ln in lines:
        m = re.search(r"\b([0-2]?\d):[0-5]\d\b", ln)
        if m:
            try:
                hr = int(m.group(1))
                if hr <= 4 or hr == 23:
                    late_night += 1
            except ValueError:
                pass

    return {
        "meta": {
            "profile_id": "MANUAL-PASTE",
            "archetype": "manual",
            "archetype_label": "Manually Pasted SMS Dump",
            "tags": ["manual"],
        },
        "bureau_data": {
            # Neutral assumptions for an unknown applicant. CIBIL is set to the
            # mid-band so the decision is driven mainly by observed cash flow.
            "cibil_score": 690,
            "active_loans": 0,
            "total_monthly_emi": 0.0,
            "hard_inquiries_30d": 0,
            "reported_gross_monthly_income": 0.0,
        },
        "device_metadata": {
            "late_night_active_hours": min(late_night, 8),
            "active_wallet_apps": 3,
            "location_variance": "medium",
            "device_age_months": 12,
            "is_rooted_device": False,
        },
        "sms_logs": lines,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="uw-header">
        <div class="uw-title">◆ Alternative Credit Underwriting Engine</div>
        <div class="uw-sub">Multi-Agent · Parallel Orchestration · Real-Time Decisioning</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 1.28], gap="large")


# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — inputs
# ─────────────────────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="uw-panel-label">Ingestion Console</div>', unsafe_allow_html=True)

    source = st.radio(
        "Data source",
        options=["Mock profile", "Paste raw SMS"],
        horizontal=True,
        label_visibility="collapsed",
    )

    payload: Dict[str, Any] | None = None

    if source == "Mock profile":
        arch = st.selectbox(
            "Select borrower archetype",
            options=ARCHETYPES,
            format_func=lambda a: ARCHETYPE_LABELS[a],
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            regen = st.button("↻ Regenerate", use_container_width=True)
        if regen:
            # Re-roll just this archetype with a fresh random seed.
            st.session_state.demo_profiles[arch] = generate_profile(arch, seed=None)
        payload = st.session_state.demo_profiles[arch]

        b = payload["bureau_data"]
        d = payload["device_metadata"]
        st.markdown(
            f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                        color:#2a3a4f;background:#fff;border:1px solid #d9d2c4;
                        border-radius:4px;padding:0.7rem 0.9rem;margin:0.4rem 0;">
            <b>{payload['meta']['profile_id']}</b><br>
            CIBIL {b['cibil_score']} · {b['active_loans']} active loan(s) ·
            EMI INR {b['total_monthly_emi']:,.0f}<br>
            {b['hard_inquiries_30d']} hard inquiry(ies)/30d ·
            {d['active_wallet_apps']} wallet apps ·
            {d['late_night_active_hours']}h late-night
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"Raw SMS feed ({len(payload['sms_logs'])} messages)", expanded=False):
            for line in payload["sms_logs"]:
                st.markdown(
                    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;"
                    f"padding:2px 0;border-bottom:1px dotted #e6e0d4;'>{line}</div>",
                    unsafe_allow_html=True,
                )

    else:  # Paste raw SMS
        st.caption(
            "Paste raw transaction SMS — one per line. Bureau fields are unknown "
            "for a cold paste, so the decision leans on observed cash flow."
        )
        raw = st.text_area(
            "Raw SMS dump",
            height=240,
            label_visibility="collapsed",
            placeholder=(
                "Salary Credited INR 85,000 via NEFT\n"
                "Debited INR 450 at Blinkit\n"
                "Your slice bill of INR 4,200 is due\n"
                "Bet365 transaction successful: INR 2,000 debited 02:14\n"
                "..."
            ),
        )
        if raw.strip():
            payload = build_payload_from_raw_sms(raw)
            st.markdown(
                f"<span class='uw-mode-pill'>{len(payload['sms_logs'])} lines parsed</span>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Engine mode controls.
    has_llm = llm_available()
    mode_label = "LLM agents available" if has_llm else "Local heuristic mode"
    st.markdown(
        f"<span class='uw-mode-pill'>● {mode_label}</span>", unsafe_allow_html=True
    )
    use_llm = False
    if has_llm:
        use_llm = st.toggle("Use LLM reasoning (Anthropic)", value=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    execute = st.button(
        "▶  Execute Underwriting Engine",
        type="primary",
        use_container_width=True,
        disabled=payload is None,
    )

    if execute and payload is not None:
        with st.spinner("Spinning up parallel agents…"):
            try:
                st.session_state.result = run_underwriting_sync(
                    payload, use_llm=(use_llm if has_llm else False)
                )
                st.session_state.active_payload = payload
            except Exception as exc:  # noqa: BLE001 - surface cleanly in UI
                st.session_state.result = None
                st.error(f"Engine error: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — agent reasoning + decision certificate
# ─────────────────────────────────────────────────────────────────────────────
_AGENT_META = {
    "agent_a": ("A", "Cash-Flow Specialist", "Income · Liabilities · Margin"),
    "agent_b": ("B", "Behavioural Risk Auditor", "Gambling · BNPL · Cash Crunch"),
    "agent_c": ("C", "Bureau Cross-Reference", "Inquiries · Consistency"),
}


def _render_agent(agent: Dict[str, Any]) -> None:
    letter, name, subtitle = _AGENT_META.get(
        agent["agent_id"], ("?", agent["agent_name"], "")
    )
    mode = agent["mode"].upper()
    dur = agent["duration_ms"]
    title = f"Agent {letter} · {name}  —  {mode} · {dur:.0f} ms"
    with st.expander(title, expanded=True):
        st.caption(subtitle)
        if agent.get("error"):
            st.warning(agent["error"], icon="⚠️")

        out = agent["output"]
        # Render scalar metrics as a tidy two-column table, lists as bullets.
        scalars = {k: v for k, v in out.items() if not isinstance(v, (list, dict))}
        lists = {k: v for k, v in out.items() if isinstance(v, list)}

        if scalars:
            df = pd.DataFrame(
                [(k.replace("_", " ").title(), _fmt(v)) for k, v in scalars.items()],
                columns=["Metric", "Value"],
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

        for key, items in lists.items():
            if items:
                st.markdown(f"**{key.replace('_', ' ').title()}**")
                for it in items:
                    st.markdown(f"- {it}")

        with st.popover("View raw JSON"):
            st.code(json.dumps(out, indent=2), language="json")


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _tier_class(tier: str) -> str:
    return {"Tier-1": "t1", "Tier-2": "t2", "Tier-3": "t3"}.get(tier, "t3")


def _dti_class(dti_pct: float) -> str:
    if dti_pct < 35:
        return "green"
    if dti_pct <= 50:
        return "amber"
    return "red"


def _render_certificate(decision: Dict[str, Any], engine_mode: str, total_ms: float) -> None:
    tier = decision["risk_tier"]
    tcls = _tier_class(tier)
    fin = decision["financials"]
    rate = decision["interest_rate_pct"]
    rate_str = f"{rate:.2f}%" if rate is not None else "—"
    loan_str = (
        f"₹{decision['max_loan_amount']:,}" if decision["max_loan_amount"] else "₹0"
    )
    status = decision["status"]
    status_color = "green" if status == "Approved" else "red"

    st.markdown(
        f"""
        <div class="uw-cert">
            <div class="uw-cert-head">
                <div>
                    <div class="uw-cert-kicker">Credit Decision Certificate</div>
                    <div class="uw-cert-title">{decision['profile_id']}</div>
                </div>
                <div style="text-align:right;">
                    <span class="uw-badge {tcls}">{tier} · {decision['risk_tier_label']}</span>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                                margin-top:0.4rem;color:var(--ink-soft);">
                        Status: <b style="color:{'#1f7a55' if status=='Approved' else '#a32f2f'}">{status}</b>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metrics as native Streamlit columns (more reliable than nested HTML divs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sanction Limit", loan_str)
    m2.metric("Interest Rate", rate_str)
    dti_delta = "Low" if decision['dti_pct'] < 35 else ("Moderate" if decision['dti_pct'] <= 50 else "High")
    m3.metric("DTI Ratio", f"{decision['dti_pct']}%", delta=dti_delta,
              delta_color="normal" if decision['dti_pct'] < 35 else "inverse")
    m4.metric("CIBIL Score", fin['cibil_score'])

    # DTI breakdown table.
    st.markdown('<div class="uw-panel-label" style="margin-top:1.1rem;">DTI Computation</div>', unsafe_allow_html=True)
    dti_df = pd.DataFrame(
        [
            ("Gross Monthly Income", f"₹{fin['gross_monthly_income']:,.0f}"),
            ("Bureau Monthly EMIs", f"₹{fin['bureau_monthly_emi']:,.0f}"),
            ("Hidden Liabilities (subs + BNPL)", f"₹{fin['hidden_liabilities']:,.0f}"),
            ("Total Monthly Obligations", f"₹{fin['total_monthly_obligations']:,.0f}"),
            ("DTI = Obligations / Income", f"{decision['dti_pct']}%"),
        ],
        columns=["Component", "Amount"],
    )
    st.dataframe(dti_df, hide_index=True, use_container_width=True)

    # Decision rationale.
    st.markdown('<div class="uw-panel-label" style="margin-top:0.8rem;">Decision Rationale</div>', unsafe_allow_html=True)
    reason_df = pd.DataFrame(
        [(f"{i+1}", r) for i, r in enumerate(decision["reasons"])],
        columns=["#", "Reason"],
    )
    st.dataframe(reason_df, hide_index=True, use_container_width=True)

    st.markdown(
        f"""
        <div class="uw-foot">
            <span>Engine mode: {engine_mode.upper()} · Base rate {BASE_INTEREST_RATE:.2f}%</span>
            <span>Pipeline {total_ms:.0f} ms</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


with right:
    st.markdown('<div class="uw-panel-label">Agent Telemetry</div>', unsafe_allow_html=True)

    result = st.session_state.result
    if result is None:
        st.info(
            "Configure a borrower on the left and run the engine. "
            "Three specialist agents will execute in parallel, then the Lead "
            "Underwriter synthesises the final credit decision.",
            icon="📋",
        )
    else:
        # Agent reasoning panels.
        for agent in result["agents"]:
            _render_agent(agent)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="uw-panel-label">Lead Underwriter Synthesis</div>', unsafe_allow_html=True)
        _render_certificate(
            result["decision"], result["engine_mode"], result["total_duration_ms"]
        )

        with st.popover("Export full decision JSON"):
            st.code(json.dumps(result, indent=2), language="json")
