"""
data_generator.py
═════════════════════════════════════════════════════════════════════════════
Synthetic borrower-profile generator for the Multi-Agent Alternative Credit
Underwriting Engine.

Produces deliberately *messy*, realistic alternative-data payloads that mimic
what an Indian digital-lending NBFC actually ingests: raw transaction SMS
strings, device telemetry, and a thin traditional-bureau snapshot.

Three archetypes are supported:
    1. "prime"      → High-earner prime borrower (clean, low risk)
    2. "thin_file"  → Cash-crunched borrower juggling multiple BNPL services
    3. "high_risk"  → Volatile profile with gambling patterns / credit hunger

The module is import-safe (no side effects on import) and can also be run
directly as a CLI:

    python data_generator.py --archetype prime --seed 7

Every profile is a plain, JSON-serialisable nested dict so it can flow
straight into the async agent engine and the Streamlit UI without adapters.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Canonical archetype identifiers
# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPE_PRIME = "prime"
ARCHETYPE_THIN_FILE = "thin_file"
ARCHETYPE_HIGH_RISK = "high_risk"

ARCHETYPES: List[str] = [ARCHETYPE_PRIME, ARCHETYPE_THIN_FILE, ARCHETYPE_HIGH_RISK]

# Human-friendly labels for the UI dropdown.
ARCHETYPE_LABELS: Dict[str, str] = {
    ARCHETYPE_PRIME: "Prime Borrower — High Earner (Low Risk)",
    ARCHETYPE_THIN_FILE: "Thin-File Borrower — BNPL Stacked (Cash Crunch)",
    ARCHETYPE_HIGH_RISK: "High-Risk Borrower — Gambling / Credit Hunger",
}

# ─────────────────────────────────────────────────────────────────────────────
# Merchant / vendor vocabularies used to synthesise SMS noise
# ─────────────────────────────────────────────────────────────────────────────
_GROCERY_MERCHANTS = ["Blinkit", "Zepto", "BigBasket", "Instamart", "DMart"]
_FOOD_MERCHANTS = ["Swiggy", "Zomato", "Dominos", "EatFit", "KFC"]
_SHOPPING_MERCHANTS = ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa"]
_UTILITY_MERCHANTS = ["Airtel", "Jio", "BESCOM", "TataPower", "ACT Fibernet"]
_BNPL_PROVIDERS = ["slice", "uni", "postpe", "LazyPay", "Simpl", "ZestMoney"]
_GAMBLING_MERCHANTS = ["Bet365", "Dream11", "MPL", "Parimatch", "RummyCircle"]
_BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak"]


@dataclass
class _ProfileSpec:
    """Internal tunable ranges that distinguish one archetype from another."""

    income_range: tuple
    cibil_range: tuple
    active_loans_range: tuple
    emi_ratio_range: tuple          # monthly EMI as a fraction of income
    hard_inquiries_range: tuple
    late_night_hours_range: tuple
    wallet_apps_range: tuple
    location_variance: List[str]
    bnpl_intensity: int             # number of distinct BNPL bills to inject
    gambling_intensity: int         # number of gambling SMS to inject
    discretionary_spend_factor: float  # multiplier on routine spend volume
    salary_is_regular: bool
    tags: List[str] = field(default_factory=list)


# Each archetype is fully described by a spec — no magic numbers scattered
# through the generation logic.
_SPECS: Dict[str, _ProfileSpec] = {
    ARCHETYPE_PRIME: _ProfileSpec(
        income_range=(85_000, 180_000),
        cibil_range=(760, 830),
        active_loans_range=(0, 1),
        emi_ratio_range=(0.05, 0.18),
        hard_inquiries_range=(0, 1),
        late_night_hours_range=(0, 2),
        wallet_apps_range=(1, 3),
        location_variance=["low", "low", "medium"],
        bnpl_intensity=0,
        gambling_intensity=0,
        discretionary_spend_factor=1.0,
        salary_is_regular=True,
        tags=["prime", "salaried"],
    ),
    ARCHETYPE_THIN_FILE: _ProfileSpec(
        income_range=(22_000, 45_000),
        cibil_range=(640, 710),
        active_loans_range=(0, 1),
        emi_ratio_range=(0.10, 0.28),
        hard_inquiries_range=(2, 5),
        late_night_hours_range=(2, 5),
        wallet_apps_range=(4, 7),
        location_variance=["medium", "medium", "high"],
        bnpl_intensity=4,
        gambling_intensity=0,
        discretionary_spend_factor=1.4,
        salary_is_regular=False,
        tags=["thin_file", "bnpl_stacked", "gig_income"],
    ),
    ARCHETYPE_HIGH_RISK: _ProfileSpec(
        income_range=(30_000, 70_000),
        cibil_range=(560, 660),
        active_loans_range=(1, 4),
        emi_ratio_range=(0.25, 0.45),
        hard_inquiries_range=(5, 11),
        late_night_hours_range=(4, 7),
        wallet_apps_range=(5, 9),
        location_variance=["high", "high", "medium"],
        bnpl_intensity=3,
        gambling_intensity=6,
        discretionary_spend_factor=1.8,
        salary_is_regular=False,
        tags=["high_risk", "gambling", "credit_hungry"],
    ),
}


def _money(value: float) -> str:
    """Format a number as an Indian-style INR string (e.g. 85,000)."""
    whole = int(round(value))
    s = str(whole)
    if len(s) <= 3:
        return s
    # Indian grouping: last 3 digits, then groups of 2.
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def _timestamp(rng: random.Random, days_back: int) -> str:
    """Produce a plausible SMS timestamp string within the last `days_back`."""
    base = datetime(2025, 6, 30, 9, 0, 0)
    delta = timedelta(
        days=rng.randint(0, days_back),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
    )
    return (base - delta).strftime("%d-%b-%Y %H:%M")


def _build_sms_logs(rng: random.Random, spec: _ProfileSpec, income: float) -> List[str]:
    """
    Assemble a shuffled list of raw, messy transaction SMS strings consistent
    with the archetype. Intentionally inconsistent formatting mimics the noise
    of real telecom SMS feeds.
    """
    logs: List[str] = []
    bank = rng.choice(_BANKS)

    # ── Income credit(s) ────────────────────────────────────────────────────
    if spec.salary_is_regular:
        logs.append(
            f"{_timestamp(rng, 3)} - {bank} Bank: Salary Credited INR "
            f"{_money(income)} via NEFT. Avl Bal: INR {_money(income * 1.6)}"
        )
    else:
        # Irregular / gig income split into uneven chunks.
        chunks = rng.randint(2, 4)
        remaining = income
        for i in range(chunks):
            part = remaining / (chunks - i) * rng.uniform(0.7, 1.3)
            part = min(part, remaining)
            remaining -= part
            src = rng.choice(["UPI", "IMPS", "NEFT", "Cash Deposit"])
            logs.append(
                f"{_timestamp(rng, 28)} {bank}: Credit of Rs {_money(part)} "
                f"received via {src}. Ref no {rng.randint(100000, 999999)}"
            )

    # ── Routine discretionary spend ──────────────────────────────────────────
    n_routine = int(8 * spec.discretionary_spend_factor)
    for _ in range(n_routine):
        category = rng.choice(
            [_GROCERY_MERCHANTS, _FOOD_MERCHANTS, _SHOPPING_MERCHANTS, _UTILITY_MERCHANTS]
        )
        merchant = rng.choice(category)
        amt = rng.choice([149, 199, 250, 320, 450, 599, 780, 1200, 1850, 2400])
        logs.append(
            f"{_timestamp(rng, 30)} Debited INR {_money(amt)} at {merchant}. "
            f"UPI Ref {rng.randint(100000000, 999999999)}"
        )

    # ── Recurring subscriptions (fixed liabilities Agent A should detect) ─────
    subs = rng.sample(
        [
            ("Netflix", 649),
            ("Spotify", 119),
            ("Amazon Prime", 299),
            ("Hotstar", 299),
            ("Cult.fit", 1300),
            ("YouTube Premium", 129),
        ],
        k=rng.randint(1, 3),
    )
    for name, amt in subs:
        logs.append(
            f"{_timestamp(rng, 30)} Auto-pay of INR {_money(amt)} to {name} "
            f"successful. Mandate ID {rng.randint(10000, 99999)}"
        )

    # ── BNPL bills (signals BNPL stacking for Agent B) ───────────────────────
    if spec.bnpl_intensity > 0:
        providers = rng.sample(_BNPL_PROVIDERS, k=min(spec.bnpl_intensity, len(_BNPL_PROVIDERS)))
        for p in providers:
            amt = rng.choice([1200, 2100, 3400, 4200, 5600, 6800])
            logs.append(
                f"{_timestamp(rng, 15)} Your {p} bill of INR {_money(amt)} "
                f"is due on {rng.randint(1, 28)}th. Pay now to avoid late fees."
            )

    # ── Gambling transactions (severe behavioural flag for Agent B) ──────────
    for _ in range(spec.gambling_intensity):
        merchant = rng.choice(_GAMBLING_MERCHANTS)
        amt = rng.choice([500, 1000, 2000, 3500, 5000])
        hour = rng.choice([1, 2, 3, 23, 0])  # late-night skew
        logs.append(
            f"30-Jun-2025 {hour:02d}:{rng.randint(10,59)} {merchant} transaction "
            f"successful: INR {_money(amt)} debited. Txn ID {rng.randint(10**6, 10**7)}"
        )

    # ── Occasional bounce / low-balance alert for riskier files ──────────────
    if not spec.salary_is_regular and rng.random() < 0.6:
        logs.append(
            f"{_timestamp(rng, 20)} ALERT: Auto-debit of INR "
            f"{_money(rng.choice([2100, 3400, 4200]))} FAILED due to "
            f"insufficient balance. Please maintain sufficient funds."
        )

    rng.shuffle(logs)
    return logs


def generate_profile(archetype: str = ARCHETYPE_PRIME, seed: int | None = None) -> Dict:
    """
    Generate a single synthetic borrower profile for the given archetype.

    Parameters
    ----------
    archetype : str
        One of ARCHETYPE_PRIME, ARCHETYPE_THIN_FILE, ARCHETYPE_HIGH_RISK.
    seed : int | None
        Optional RNG seed for reproducible profiles. If None, a random seed
        is used and recorded in the returned profile under `meta.seed`.

    Returns
    -------
    dict
        A fully JSON-serialisable nested profile dictionary.
    """
    if archetype not in _SPECS:
        raise ValueError(
            f"Unknown archetype '{archetype}'. "
            f"Expected one of: {', '.join(ARCHETYPES)}"
        )

    if seed is None:
        seed = random.randint(1, 10_000_000)
    rng = random.Random(seed)
    spec = _SPECS[archetype]

    income = rng.randint(*spec.income_range)
    cibil = rng.randint(*spec.cibil_range)
    active_loans = rng.randint(*spec.active_loans_range)
    emi_ratio = rng.uniform(*spec.emi_ratio_range)
    monthly_emi = round(income * emi_ratio, 0) if active_loans > 0 else 0.0
    hard_inquiries = rng.randint(*spec.hard_inquiries_range)

    profile = {
        "meta": {
            "profile_id": f"BORR-{rng.randint(10**5, 10**6 - 1)}",
            "archetype": archetype,
            "archetype_label": ARCHETYPE_LABELS[archetype],
            "seed": seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tags": list(spec.tags),
        },
        "bureau_data": {
            "cibil_score": cibil,
            "active_loans": active_loans,
            "total_monthly_emi": float(monthly_emi),
            "hard_inquiries_30d": hard_inquiries,
            "reported_gross_monthly_income": float(income),
        },
        "device_metadata": {
            "late_night_active_hours": rng.randint(*spec.late_night_hours_range),
            "active_wallet_apps": rng.randint(*spec.wallet_apps_range),
            "location_variance": rng.choice(spec.location_variance),
            "device_age_months": rng.randint(2, 60),
            "is_rooted_device": rng.random() < (0.25 if archetype == ARCHETYPE_HIGH_RISK else 0.03),
        },
        "sms_logs": _build_sms_logs(rng, spec, income),
    }
    return profile


def generate_all_profiles(seed: int | None = 42) -> Dict[str, Dict]:
    """
    Generate one profile per archetype with deterministic, offset seeds so the
    demo set is stable across reruns (useful for the Streamlit dropdown).
    """
    base = 42 if seed is None else seed
    return {
        ARCHETYPE_PRIME: generate_profile(ARCHETYPE_PRIME, seed=base + 1),
        ARCHETYPE_THIN_FILE: generate_profile(ARCHETYPE_THIN_FILE, seed=base + 2),
        ARCHETYPE_HIGH_RISK: generate_profile(ARCHETYPE_HIGH_RISK, seed=base + 3),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic borrower profile."
    )
    parser.add_argument(
        "--archetype",
        choices=ARCHETYPES,
        default=ARCHETYPE_PRIME,
        help="Borrower archetype to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional RNG seed for reproducibility."
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate one profile per archetype."
    )
    args = parser.parse_args()

    if args.all:
        output = generate_all_profiles(seed=args.seed)
    else:
        output = generate_profile(args.archetype, seed=args.seed)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
