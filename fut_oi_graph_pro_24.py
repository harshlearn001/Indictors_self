"""
MARKETFORGE FUTURES OI ANALYZER PRO 2.4
Dual-Model Positioning Intelligence

Data source:
    MarketForge/data/master/Futures_master_three_expiries/FUTSTK
    MarketForge/data/master/Futures_master_three_expiries/FUTIDX

Contract identity:
    SYMBOL + EXP_DATE

Important:
    - Fresh positioning and position closure are scored separately.
    - Scores are rule-based research measures, not return probabilities.
    - Fixed-pair OI composition is not verified exchange rollover volume.
"""

import os
import math
from pathlib import Path
import os
import math
import string
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "2.4"
SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_marketforge_root():
    """
    Locate the MarketForge project without assuming a fixed drive letter.

    Priority:
      1. MARKETFORGE_ROOT environment variable
      2. MarketForge beside this script or in its parent folders
      3. MarketForge in the current working directory or its parents
      4. MarketForge at the root of any available Windows drive
    """
    candidates = []

    configured_root = os.environ.get("MARKETFORGE_ROOT", "").strip()
    if configured_root:
        candidates.append(Path(configured_root).expanduser())

    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents, Path.cwd(), *Path.cwd().parents):
        candidates.append(parent / "MarketForge")

    # Search drive roots so this works whether the project is on H:, D:, E:, etc.
    if os.name == "nt":
        for drive in string.ascii_uppercase:
            candidates.append(Path(f"{drive}:\\") / "MarketForge")

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)

        data_root = candidate / "data" / "master" / "Futures_master_three_expiries"
        if (data_root / "FUTSTK").is_dir() and (data_root / "FUTIDX").is_dir():
            return candidate

    checked = "\\n".join(f"  - {item}" for item in candidates[:12])
    raise FileNotFoundError(
        "Could not locate the MarketForge project.\\n"
        "Set MARKETFORGE_ROOT to the folder containing the data directory.\\n"
        "Example (PowerShell): $env:MARKETFORGE_ROOT = 'H:\\\\MarketForge'\\n"
        "Some locations checked:\\n" + checked
    )


MARKETFORGE_ROOT = resolve_marketforge_root()

DATA_ROOT = (
    MARKETFORGE_ROOT
    / "data"
    / "master"
    / "Futures_master_three_expiries"
)
FUTSTK_PATH = DATA_ROOT / "FUTSTK"
FUTIDX_PATH = DATA_ROOT / "FUTIDX"

OUTPUT_PATH = (
    MARKETFORGE_ROOT
    / "output"
    / "futures_oi_analyzer"
)

LOOKBACK_SESSIONS = 25
EXPIRIES = ("NEAR", "NEXT", "FAR")

TIMEFRAME_WEIGHTS = {1: 0.50, 3: 0.30, 5: 0.20}
EXPIRY_WEIGHTS = {"NEAR": 0.50, "NEXT": 0.30, "FAR": 0.20}

SAVE_CHARTS = True
SHOW_CHARTS = True
EXPORT_CSV = True
EXPORT_EXCEL = True

REQUIRED_COLUMNS = [
    "TRADE_DATE", "INSTRUMENT", "SYMBOL", "EXP_DATE",
    "CLOSE_PRICE", "OPEN_INT", "EXPIRY_TYPE"
]
OPTIONAL_NUMERIC_COLUMNS = [
    "OPEN_PRICE", "HI_PRICE", "LO_PRICE", "TRD_VAL",
    "TRD_QTY", "NO_OF_CONT", "NO_OF_TRADE"
]

SIGNAL_COLORS = {
    "LONG_BUILDUP": "#087f5b",
    "SHORT_BUILDUP": "#ff4d6d",
    "SHORT_COVERING": "#66c2a5",
    "LONG_UNWINDING": "#f6b44b",
    "NEUTRAL": "#9ca3af",
}
PRICE_COLOR = "#2478bd"
TRANSITION_COLOR = "#8b5cf6"


# ============================================================
# GENERAL HELPERS
# ============================================================

def parse_dates(series):
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(text, errors="coerce")
    return parsed.fillna(fallback)


def pct_change_from_values(current, previous):
    """Safe percentage change; undefined when prior value is zero."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return (current / previous - 1.0) * 100.0


def fmt_pct(value):
    return "N/A" if pd.isna(value) else f"{value:+.2f}%"


def fmt_num(value):
    return "N/A" if pd.isna(value) else f"{value:,.0f}"


def fmt_score(value):
    return "N/A" if pd.isna(value) else f"{value:.1f}"


def weighted_average(values, weights):
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if pd.notna(v) and w > 0
    ]
    if not pairs:
        return np.nan
    total_w = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_w


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_data(symbol):
    stock_file = FUTSTK_PATH / f"{symbol}.csv"
    index_file = FUTIDX_PATH / f"{symbol}.csv"

    if stock_file.is_file():
        file_path, instrument_group = stock_file, "STOCK"
    elif index_file.is_file():
        file_path, instrument_group = index_file, "INDEX"
    else:
        print(f"\nERROR: {symbol} was not found in FUTSTK or FUTIDX.")
        print(f"Checked: {FUTSTK_PATH}")
        print(f"         {FUTIDX_PATH}")
        return pd.DataFrame()

    print(f"\n{instrument_group} DATA: {symbol}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.astype(str).str.strip().str.upper()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["TRADE_DATE"] = parse_dates(df["TRADE_DATE"])
    df["EXP_DATE"] = parse_dates(df["EXP_DATE"])

    for col in ["SYMBOL", "INSTRUMENT", "EXPIRY_TYPE"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    numeric_columns = ["CLOSE_PRICE", "OPEN_INT"] + OPTIONAL_NUMERIC_COLUMNS
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )

    df = df[df["SYMBOL"] == symbol].copy()
    df = df.dropna(
        subset=["TRADE_DATE", "EXP_DATE", "CLOSE_PRICE", "OPEN_INT"]
    )
    df = df[(df["CLOSE_PRICE"] > 0) & (df["OPEN_INT"] >= 0)].copy()
    df = df[df["EXPIRY_TYPE"].isin(EXPIRIES)].copy()

    duplicate_contracts = int(
        df.duplicated(["TRADE_DATE", "SYMBOL", "EXP_DATE"]).sum()
    )
    duplicate_buckets = int(
        df.duplicated(["TRADE_DATE", "EXPIRY_TYPE"]).sum()
    )

    if duplicate_contracts:
        raise ValueError(
            f"{duplicate_contracts} duplicate SYMBOL/EXP_DATE/date records. "
            "Resolve duplicates before analysis."
        )
    if duplicate_buckets:
        raise ValueError(
            f"{duplicate_buckets} duplicate date/expiry-bucket records. "
            "Expected at most one active contract per expiry bucket per date."
        )

    df = df.sort_values(
        ["TRADE_DATE", "EXP_DATE"]
    ).reset_index(drop=True)

    if df.empty:
        print("No valid records after cleaning.")
        return df

    print(f"Loaded records: {len(df):,}")
    print(f"Date range: {df.TRADE_DATE.min().date()} to {df.TRADE_DATE.max().date()}")
    print(f"Duplicate contract records: {duplicate_contracts}")
    print(f"Duplicate expiry-bucket records: {duplicate_buckets}")
    print(f"Data directory: {DATA_ROOT}")
    return df


# ============================================================
# CURRENT CONTRACT DISCOVERY
# ============================================================

def get_active_contracts(df):
    latest_date = df["TRADE_DATE"].max()
    latest = df[df["TRADE_DATE"] == latest_date].copy()
    latest = latest.sort_values("EXP_DATE")

    contracts = []
    print("\nLATEST ACTIVE CONTRACTS")
    print("-" * 88)

    for _, row in latest.iterrows():
        item = {
            "SYMBOL": row["SYMBOL"],
            "EXPIRY_TYPE": row["EXPIRY_TYPE"],
            "EXP_DATE": row["EXP_DATE"],
            "PRICE": float(row["CLOSE_PRICE"]),
            "OI": float(row["OPEN_INT"]),
        }
        contracts.append(item)
        print(
            f"{item['EXPIRY_TYPE']:5s} | Expiry: {item['EXP_DATE'].date()} "
            f"| Price: {item['PRICE']:,.2f} | OI: {item['OI']:,.0f}"
        )

    return contracts


# ============================================================
# PRICE/OI SIGNALS
# ============================================================

def classify_signal(price_pct, oi_pct):
    if pd.isna(price_pct) or pd.isna(oi_pct):
        return "NEUTRAL"
    if price_pct > 0 and oi_pct > 0:
        return "LONG_BUILDUP"
    if price_pct < 0 and oi_pct > 0:
        return "SHORT_BUILDUP"
    if price_pct > 0 and oi_pct < 0:
        return "SHORT_COVERING"
    if price_pct < 0 and oi_pct < 0:
        return "LONG_UNWINDING"
    return "NEUTRAL"


def add_contract_metrics(history):
    history = history.sort_values("TRADE_DATE").reset_index(drop=True).copy()

    history["OI_CHANGE"] = history["OPEN_INT"].diff()
    history["PRICE_CHANGE"] = history["CLOSE_PRICE"].diff()

    history["OI_1D_PCT"] = history["OPEN_INT"].pct_change(fill_method=None) * 100
    history["PRICE_1D_PCT"] = history["CLOSE_PRICE"].pct_change(fill_method=None) * 100

    for period in (3, 5, 10):
        history[f"OI_{period}D_CHANGE"] = history["OPEN_INT"].diff(period)
        history[f"PRICE_{period}D_CHANGE"] = history["CLOSE_PRICE"].diff(period)
        history[f"OI_{period}D_PCT"] = (
            history["OPEN_INT"].pct_change(periods=period, fill_method=None) * 100
        )
        history[f"PRICE_{period}D_PCT"] = (
            history["CLOSE_PRICE"].pct_change(periods=period, fill_method=None) * 100
        )

    for period in (1, 3, 5):
        oi_col = "OI_1D_PCT" if period == 1 else f"OI_{period}D_PCT"
        price_col = "PRICE_1D_PCT" if period == 1 else f"PRICE_{period}D_PCT"
        history[f"SIGNAL_{period}D"] = [
            classify_signal(p, o)
            for p, o in zip(history[price_col], history[oi_col])
        ]

    # A change in rolling 3-session momentum, expressed in percentage points.
    history["OI_MOMENTUM_CHANGE_3D_PP"] = (
        history["OI_3D_PCT"] - history["OI_3D_PCT"].shift(3)
    )
    history["OI_MOMENTUM_CHANGE_5D_PP"] = (
        history["OI_5D_PCT"] - history["OI_5D_PCT"].shift(5)
    )
    history["PRICE_MOMENTUM_CHANGE_3D_PP"] = (
        history["PRICE_3D_PCT"] - history["PRICE_3D_PCT"].shift(3)
    )

    # OI expansion score: 50 is neutral; above 50 = expansion,
    # below 50 = contraction. This is not bullish/bearish direction.
    def oi_expansion_score(row):
        value = weighted_average(
            [row["OI_1D_PCT"], row["OI_3D_PCT"], row["OI_5D_PCT"]],
            [0.50, 0.30, 0.20]
        )
        if pd.isna(value):
            return np.nan
        return float(np.clip(50 + 50 * math.tanh(value / 25.0), 0, 100))

    history["OI_EXPANSION_SCORE"] = history.apply(oi_expansion_score, axis=1)

    # Role changes are descriptive bucket transitions, not rollover volume.
    history["PREVIOUS_EXPIRY_TYPE"] = history["EXPIRY_TYPE"].shift(1)
    history["ROLE_TRANSITION"] = (
        history["PREVIOUS_EXPIRY_TYPE"].notna()
        & history["EXPIRY_TYPE"].ne(history["PREVIOUS_EXPIRY_TYPE"])
    )

    return history


# ============================================================
# DUAL-MODEL SCORING
# ============================================================

SIGNAL_TO_COMPONENT = {
    "LONG_BUILDUP": "FRESH_BULLISH",
    "SHORT_BUILDUP": "FRESH_BEARISH",
    "SHORT_COVERING": "BULLISH_CLOSURE",
    "LONG_UNWINDING": "BEARISH_CLOSURE",
    "NEUTRAL": None,
}


def score_contract(history):
    """
    Each component is a weighted percentage of valid 1D/3D/5D
    observations showing that signal family.

    Four components are deliberately independent:
      FRESH_BULLISH: new long buildup
      FRESH_BEARISH: new short buildup
      BULLISH_CLOSURE: short covering
      BEARISH_CLOSURE: long unwinding
    """
    if history.empty:
        return None

    latest = history.iloc[-1]
    components = {
        "FRESH_BULLISH": [],
        "FRESH_BEARISH": [],
        "BULLISH_CLOSURE": [],
        "BEARISH_CLOSURE": [],
    }
    valid_weights = []

    for period, weight in TIMEFRAME_WEIGHTS.items():
        signal = latest[f"SIGNAL_{period}D"]

        # A neutral signal can mean either true neutral or insufficient history.
        oi_col = "OI_1D_PCT" if period == 1 else f"OI_{period}D_PCT"
        px_col = "PRICE_1D_PCT" if period == 1 else f"PRICE_{period}D_PCT"

        if pd.isna(latest[oi_col]) or pd.isna(latest[px_col]):
            continue

        valid_weights.append(weight)
        component = SIGNAL_TO_COMPONENT[signal]
        if component is not None:
            components[component].append(weight)

    denominator = sum(valid_weights)
    if denominator <= 0:
        scores = {name: np.nan for name in components}
    else:
        scores = {
            name: round(100.0 * sum(weights) / denominator, 2)
            for name, weights in components.items()
        }

    # Net directional pressure treats short covering as positive price pressure,
    # and long unwinding as negative price pressure, but keeps those components
    # separate from fresh positioning above.
    fresh_bull = scores["FRESH_BULLISH"]
    fresh_bear = scores["FRESH_BEARISH"]
    bull_close = scores["BULLISH_CLOSURE"]
    bear_close = scores["BEARISH_CLOSURE"]

    if any(pd.isna(x) for x in [fresh_bull, fresh_bear, bull_close, bear_close]):
        net_score = np.nan
    else:
        bullish_pressure = fresh_bull + bull_close
        bearish_pressure = fresh_bear + bear_close
        # Map net pressure to a 0-100 scale; 50 = balanced.
        net_score = round(
            float(np.clip(50 + (bullish_pressure - bearish_pressure) / 2, 0, 100)),
            2
        )

    return {
        **scores,
        "NET_POSITIONING_SCORE": net_score,
        "OI_EXPANSION_SCORE": latest["OI_EXPANSION_SCORE"],
        "OI_MOMENTUM_CHANGE_3D_PP": latest["OI_MOMENTUM_CHANGE_3D_PP"],
        "OI_MOMENTUM_CHANGE_5D_PP": latest["OI_MOMENTUM_CHANGE_5D_PP"],
        "VALID_TIMEFRAME_WEIGHT": round(denominator, 3),
    }


def classify_contract_bias(net_score):
    if pd.isna(net_score):
        return "INSUFFICIENT DATA"
    if net_score >= 60:
        return "BULLISH PRESSURE"
    if net_score <= 40:
        return "BEARISH PRESSURE"
    return "BALANCED / MIXED"


def build_contract_summary(history, expiry_type):
    if history.empty:
        return None

    latest = history.iloc[-1]
    scores = score_contract(history)
    if scores is None:
        return None

    return {
        "EXPIRY_TYPE": expiry_type,
        "CONTRACT_ID": latest["CONTRACT_ID"],
        "TRADE_DATE": latest["TRADE_DATE"],
        "EXP_DATE": latest["EXP_DATE"],
        "PRICE": latest["CLOSE_PRICE"],
        "PRICE_1D_PCT": latest["PRICE_1D_PCT"],
        "OI": latest["OPEN_INT"],
        "OI_CHANGE": latest["OI_CHANGE"],
        "OI_1D_PCT": latest["OI_1D_PCT"],
        "OI_3D_PCT": latest["OI_3D_PCT"],
        "OI_5D_PCT": latest["OI_5D_PCT"],
        "OI_10D_PCT": latest["OI_10D_PCT"],
        "PRICE_3D_PCT": latest["PRICE_3D_PCT"],
        "PRICE_5D_PCT": latest["PRICE_5D_PCT"],
        "SIGNAL_1D": latest["SIGNAL_1D"],
        "SIGNAL_3D": latest["SIGNAL_3D"],
        "SIGNAL_5D": latest["SIGNAL_5D"],
        **scores,
        "CONTRACT_BIAS": classify_contract_bias(scores["NET_POSITIONING_SCORE"]),
        "ROLE_TRANSITIONS": int(history["ROLE_TRANSITION"].sum()),
        "AVAILABLE_SESSIONS": len(history),
        "TRD_QTY": latest.get("TRD_QTY", np.nan),
        "TRD_VAL": latest.get("TRD_VAL", np.nan),
        "NO_OF_CONT": latest.get("NO_OF_CONT", np.nan),
    }


# ============================================================
# MULTI-EXPIRY AGGREGATION AND CONFIDENCE
# ============================================================

SCORE_FIELDS = [
    "FRESH_BULLISH",
    "FRESH_BEARISH",
    "BULLISH_CLOSURE",
    "BEARISH_CLOSURE",
    "NET_POSITIONING_SCORE",
    "OI_EXPANSION_SCORE",
    "OI_MOMENTUM_CHANGE_3D_PP",
    "OI_MOMENTUM_CHANGE_5D_PP",
]


def aggregate_market_scores(summaries):
    valid = [
        s for s in summaries
        if s is not None and s["EXPIRY_TYPE"] in EXPIRY_WEIGHTS
    ]

    aggregate = {}
    for field in SCORE_FIELDS:
        vals, weights = [], []
        for item in valid:
            value = item.get(field, np.nan)
            if pd.notna(value):
                vals.append(value)
                weights.append(EXPIRY_WEIGHTS[item["EXPIRY_TYPE"]])
        aggregate[field] = round(weighted_average(vals, weights), 2) if vals else np.nan

    # Agreement is measured from valid expiry net scores, not from a claim
    # about the probability of future price direction.
    directional = []
    for item in valid:
        score = item.get("NET_POSITIONING_SCORE", np.nan)
        if pd.isna(score):
            continue
        if score >= 60:
            direction = "BULLISH"
        elif score <= 40:
            direction = "BEARISH"
        else:
            direction = "MIXED"
        directional.append((direction, EXPIRY_WEIGHTS[item["EXPIRY_TYPE"]]))

    net = aggregate["NET_POSITIONING_SCORE"]
    if pd.isna(net):
        overall = "INSUFFICIENT DATA"
    elif net >= 60:
        overall = "BULLISH PRESSURE"
    elif net <= 40:
        overall = "BEARISH PRESSURE"
    else:
        overall = "MIXED / BALANCED"

    if directional:
        dominant = "BULLISH" if overall == "BULLISH PRESSURE" else (
            "BEARISH" if overall == "BEARISH PRESSURE" else "MIXED"
        )
        total_w = sum(w for _, w in directional)
        agreement = sum(w for d, w in directional if d == dominant) / total_w
    else:
        agreement = 0.0

    valid_expiries = len(directional)
    data_coverage = valid_expiries / 3.0

    strength = (
        0.0 if pd.isna(net)
        else min(1.0, abs(net - 50.0) / 50.0)
    )

    # Confidence is a rule-based quality/consistency index, not probability.
    confidence_score = round(
        100.0 * (0.50 * agreement + 0.30 * strength + 0.20 * data_coverage),
        1
    ) if directional else 0.0

    if valid_expiries < 2:
        confidence_label = "LOW"
    elif confidence_score >= 75 and agreement >= 0.75:
        confidence_label = "STRONG"
    elif confidence_score >= 55 and agreement >= 0.50:
        confidence_label = "MODERATE"
    else:
        confidence_label = "LOW"

    aggregate.update({
        "OVERALL_POSITIONING": overall,
        "EXPIRY_AGREEMENT_PCT": round(agreement * 100, 1),
        "VALID_EXPIRIES": valid_expiries,
        "DATA_COVERAGE_PCT": round(data_coverage * 100, 1),
        "CONFIDENCE_SCORE": confidence_score,
        "CONFIDENCE_LABEL": confidence_label,
    })
    return aggregate


# ============================================================
# FIXED CURRENT NEAR/NEXT CONTRACT PAIR
# ============================================================

def build_fixed_pair(df, near_expiry, next_expiry):
    pair = df[df["EXP_DATE"].isin([near_expiry, next_expiry])].copy()
    if pair.empty:
        return pd.DataFrame()

    oi = pair.pivot(index="TRADE_DATE", columns="EXP_DATE", values="OPEN_INT")
    price = pair.pivot(index="TRADE_DATE", columns="EXP_DATE", values="CLOSE_PRICE")

    if near_expiry not in oi.columns or next_expiry not in oi.columns:
        return pd.DataFrame()

    oi = oi.dropna(subset=[near_expiry, next_expiry])
    if oi.empty:
        return pd.DataFrame()

    price = price.reindex(oi.index)
    result = pd.DataFrame({
        "TRADE_DATE": oi.index,
        "NEAR_OI": oi[near_expiry].values,
        "NEXT_OI": oi[next_expiry].values,
        "NEAR_PRICE": price[near_expiry].values,
        "NEXT_PRICE": price[next_expiry].values,
    })

    result["COMBINED_OI"] = result["NEAR_OI"] + result["NEXT_OI"]
    result["NEAR_OI_SHARE_PCT"] = np.where(
        result["COMBINED_OI"] > 0,
        result["NEAR_OI"] / result["COMBINED_OI"] * 100,
        np.nan
    )
    result["NEXT_OI_SHARE_PCT"] = np.where(
        result["COMBINED_OI"] > 0,
        result["NEXT_OI"] / result["COMBINED_OI"] * 100,
        np.nan
    )
    result["NEXT_SHARE_CHANGE_PP"] = result["NEXT_OI_SHARE_PCT"].diff()
    result["NEXT_PREMIUM_PCT"] = (
        result["NEXT_PRICE"] / result["NEAR_PRICE"] - 1
    ) * 100
    result["NEAR_EXP_DATE"] = near_expiry
    result["NEXT_EXP_DATE"] = next_expiry

    # Verify the share identity to catch calculation/data issues.
    share_sum = (
        result["NEAR_OI_SHARE_PCT"] + result["NEXT_OI_SHARE_PCT"]
    )
    if not np.allclose(share_sum.dropna(), 100.0, atol=1e-8):
        raise AssertionError("Fixed-pair OI shares do not sum to 100%.")

    return result.tail(LOOKBACK_SESSIONS).reset_index(drop=True)


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_reports(summaries, market):
    print("\n" + "=" * 104)
    print(f"MARKETFORGE FUTURES OI ANALYZER PRO {VERSION}")
    print("DUAL-MODEL POSITIONING INTELLIGENCE")
    print("=" * 104)

    for s in summaries:
        if s is None:
            continue
        print(f"\n{s['EXPIRY_TYPE']} CONTRACT | {s['CONTRACT_ID']}")
        print("-" * 88)
        print(f"Trade Date: {s['TRADE_DATE'].date()}")
        print(f"Expiry Date: {s['EXP_DATE'].date()}")
        print(f"Price: {s['PRICE']:,.2f} ({fmt_pct(s['PRICE_1D_PCT'])})")
        print(f"OI: {fmt_num(s['OI'])} | Daily OI: {fmt_pct(s['OI_1D_PCT'])}")
        print(
            f"OI 3D: {fmt_pct(s['OI_3D_PCT'])} | "
            f"OI 5D: {fmt_pct(s['OI_5D_PCT'])} | "
            f"OI 10D: {fmt_pct(s['OI_10D_PCT'])}"
        )
        print("DIVERGENCE:")
        print(f"  1D: {s['SIGNAL_1D']} | 3D: {s['SIGNAL_3D']} | 5D: {s['SIGNAL_5D']}")
        print("DUAL-MODEL SCORES (0-100):")
        print(f"  Fresh bullish:       {fmt_score(s['FRESH_BULLISH'])}")
        print(f"  Fresh bearish:       {fmt_score(s['FRESH_BEARISH'])}")
        print(f"  Bullish closure:     {fmt_score(s['BULLISH_CLOSURE'])}")
        print(f"  Bearish closure:     {fmt_score(s['BEARISH_CLOSURE'])}")
        print(f"  Net positioning:     {fmt_score(s['NET_POSITIONING_SCORE'])}")
        print(f"  OI expansion:        {fmt_score(s['OI_EXPANSION_SCORE'])}")
        print(f"  OI momentum change 3D: {fmt_pct(s['OI_MOMENTUM_CHANGE_3D_PP'])} pp")
        print(f"  Contract bias:       {s['CONTRACT_BIAS']}")
        print(f"  Available sessions:  {s['AVAILABLE_SESSIONS']}")

    print("\n" + "=" * 104)
    print("MULTI-EXPIRY MARKET INTELLIGENCE")
    print("=" * 104)
    print(f"Fresh bullish score:       {fmt_score(market['FRESH_BULLISH'])}")
    print(f"Fresh bearish score:       {fmt_score(market['FRESH_BEARISH'])}")
    print(f"Bullish closure score:     {fmt_score(market['BULLISH_CLOSURE'])}")
    print(f"Bearish closure score:     {fmt_score(market['BEARISH_CLOSURE'])}")
    print(f"Net positioning score:     {fmt_score(market['NET_POSITIONING_SCORE'])}")
    print(f"OI expansion score:        {fmt_score(market['OI_EXPANSION_SCORE'])}")
    print(f"Expiry agreement:          {market['EXPIRY_AGREEMENT_PCT']:.1f}%")
    print(f"Valid expiries:            {market['VALID_EXPIRIES']}/3")
    print(f"Data coverage:             {market['DATA_COVERAGE_PCT']:.1f}%")
    print(f"Overall positioning:       {market['OVERALL_POSITIONING']}")
    print(
        f"Confidence index:          {market['CONFIDENCE_SCORE']:.1f}/100 "
        f"({market['CONFIDENCE_LABEL']})"
    )
    print("\nInterpretation:")
    print("- Fresh bullish/bearish scores measure new-positioning signal frequency.")
    print("- Bullish/bearish closure scores measure covering/unwinding separately.")
    print("- Net positioning is a rule-based directional pressure index.")
    print("- Confidence is consistency/data coverage, NOT probability of profit.")
    print("- OI share is a composition proxy, NOT verified exchange rollover volume.")


# ============================================================
# DASHBOARD CHARTS
# ============================================================

def plot_contract_axis(ax, history, summary):
    if history.empty or summary is None:
        ax.set_title("No contract history")
        return

    view = history.tail(LOOKBACK_SESSIONS).reset_index(drop=True)
    x = np.arange(len(view))
    colors = view["SIGNAL_1D"].map(SIGNAL_COLORS).fillna(SIGNAL_COLORS["NEUTRAL"])

    ax.bar(x, view["OPEN_INT"] / 1e6, color=colors, width=0.78, alpha=0.88)
    ax.set_ylabel("OI (Millions)")
    ax.set_title(
        f"{summary['EXPIRY_TYPE']} | Expiry: "
        f"{summary['EXP_DATE'].strftime('%d-%b-%Y')}"
    )
    ax.grid(axis="y", alpha=0.22)

    ax2 = ax.twinx()
    ax2.plot(
        x, view["CLOSE_PRICE"],
        color=PRICE_COLOR, linewidth=1.9, marker="o", markersize=2.5
    )
    ax2.set_ylabel("Futures Price")

    step = max(len(view) // 10, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(
        view["TRADE_DATE"].dt.strftime("%d-%b").iloc[::step],
        rotation=35, ha="right"
    )

    for i, row in view.iterrows():
        if bool(row["ROLE_TRANSITION"]):
            ax.axvline(i, color=TRANSITION_COLOR, linestyle="--", linewidth=1)


def plot_pair(ax, pair):
    if pair.empty:
        ax.set_title("Fixed near/next pair history unavailable")
        return

    ax.plot(
        pair["TRADE_DATE"], pair["NEAR_OI_SHARE_PCT"],
        color="#0891b2", marker="o", markersize=3, linewidth=1.8,
        label="Near OI share"
    )
    ax.plot(
        pair["TRADE_DATE"], pair["NEXT_OI_SHARE_PCT"],
        color="#8b5cf6", marker="o", markersize=3, linewidth=1.8,
        label="Next OI share"
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("OI composition (%)")
    ax.set_title("FIXED NEAR / NEXT CONTRACT OI COMPOSITION")
    ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8, loc="best")


def plot_heatmap(ax, summaries):
    fields = [
        "FRESH_BULLISH", "FRESH_BEARISH",
        "BULLISH_CLOSURE", "BEARISH_CLOSURE",
        "NET_POSITIONING_SCORE", "OI_EXPANSION_SCORE"
    ]
    labels = [
        "Fresh Bull", "Fresh Bear", "Bull Closure",
        "Bear Closure", "Net Position", "OI Expansion"
    ]
    rows = []
    row_labels = []
    for expiry in EXPIRIES:
        item = next(
            (s for s in summaries if s and s["EXPIRY_TYPE"] == expiry),
            None
        )
        if item is None:
            continue
        rows.append([item.get(f, np.nan) for f in fields])
        row_labels.append(expiry)

    if not rows:
        ax.axis("off")
        return

    matrix = np.array(rows, dtype=float)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("EXPIRY POSITIONING HEATMAP")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "N/A" if np.isnan(value) else f"{value:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Score (0-100)")


def plot_scorecard(ax, summaries, market):
    ax.axis("off")
    headers = [
        "EXPIRY", "FRESH BULL", "FRESH BEAR",
        "BULL CLOSURE", "BEAR CLOSURE", "NET", "BIAS"
    ]
    rows = []
    for s in summaries:
        if s is None:
            continue
        rows.append([
            s["EXPIRY_TYPE"],
            fmt_score(s["FRESH_BULLISH"]),
            fmt_score(s["FRESH_BEARISH"]),
            fmt_score(s["BULLISH_CLOSURE"]),
            fmt_score(s["BEARISH_CLOSURE"]),
            fmt_score(s["NET_POSITIONING_SCORE"]),
            s["CONTRACT_BIAS"]
        ])
    rows.append([
        "OVERALL",
        fmt_score(market["FRESH_BULLISH"]),
        fmt_score(market["FRESH_BEARISH"]),
        fmt_score(market["BULLISH_CLOSURE"]),
        fmt_score(market["BEARISH_CLOSURE"]),
        fmt_score(market["NET_POSITIONING_SCORE"]),
        market["OVERALL_POSITIONING"]
    ])
    table = ax.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.5)
    ax.set_title(
        f"DUAL-MODEL SCORECARD | Confidence: "
        f"{market['CONFIDENCE_LABEL']} ({market['CONFIDENCE_SCORE']:.1f}/100)",
        fontweight="bold", pad=9
    )


def plot_latest_summary(ax, summaries):
    ax.axis("off")
    headers = [
        "EXPIRY", "EXP DATE", "PRICE", "PRICE %",
        "OI (M)", "OI %", "OI 5D %", "1D SIGNAL"
    ]
    rows = []
    for s in summaries:
        if s is None:
            continue
        rows.append([
            s["EXPIRY_TYPE"],
            s["EXP_DATE"].strftime("%d-%b-%Y"),
            f"{s['PRICE']:,.2f}",
            fmt_pct(s["PRICE_1D_PCT"]),
            f"{s['OI'] / 1e6:.2f}",
            fmt_pct(s["OI_1D_PCT"]),
            fmt_pct(s["OI_5D_PCT"]),
            s["SIGNAL_1D"]
        ])
    table = ax.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.45)
    ax.set_title("LATEST CONTRACT SUMMARY", fontweight="bold", pad=8)


def create_dashboard(symbol, histories, summaries, pair, market):
    fig = plt.figure(figsize=(17, 20), constrained_layout=True)
    grid = fig.add_gridspec(
        7, 1,
        height_ratios=[1, 1, 1, 0.8, 0.9, 0.8, 0.7]
    )
    fig.suptitle(
        f"MARKETFORGE FUTURES OI ANALYZER PRO {VERSION} | {symbol}",
        fontsize=15, fontweight="bold"
    )

    for idx, expiry in enumerate(EXPIRIES):
        ax = fig.add_subplot(grid[idx])
        history = histories.get(expiry, pd.DataFrame())
        summary = next(
            (s for s in summaries if s and s["EXPIRY_TYPE"] == expiry),
            None
        )
        plot_contract_axis(ax, history, summary)

    plot_pair(fig.add_subplot(grid[3]), pair)
    plot_heatmap(fig.add_subplot(grid[4]), summaries)
    plot_scorecard(fig.add_subplot(grid[5]), summaries, market)
    plot_latest_summary(fig.add_subplot(grid[6]), summaries)

    legend = [
        Patch(color=SIGNAL_COLORS["LONG_BUILDUP"], label="Long Buildup"),
        Patch(color=SIGNAL_COLORS["SHORT_BUILDUP"], label="Short Buildup"),
        Patch(color=SIGNAL_COLORS["SHORT_COVERING"], label="Short Covering"),
        Patch(color=SIGNAL_COLORS["LONG_UNWINDING"], label="Long Unwinding"),
        Patch(color=SIGNAL_COLORS["NEUTRAL"], label="Neutral"),
        Line2D([0], [0], color=PRICE_COLOR, lw=2, label="Futures Price"),
        Line2D([0], [0], color=TRANSITION_COLOR, ls="--", label="Expiry Role Change")
    ]
    fig.legend(
        handles=legend, loc="lower center",
        ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.005)
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / f"{symbol}_FUTURES_OI_PRO_24.png"

    if SAVE_CHARTS:
        fig.savefig(output_file, dpi=160, bbox_inches="tight")
        print(f"\nDashboard saved: {output_file}")

    if SHOW_CHARTS:
        plt.show()
    plt.close(fig)


# ============================================================
# EXPORTS
# ============================================================

def export_results(symbol, histories, summaries, pair, market):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    history_frames = [
        frame.copy() for frame in histories.values() if not frame.empty
    ]
    history_df = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames else pd.DataFrame()
    )
    summary_df = pd.DataFrame([s for s in summaries if s is not None])
    market_df = pd.DataFrame([market])

    if EXPORT_CSV:
        if not history_df.empty:
            history_df.to_csv(
                OUTPUT_PATH / f"{symbol}_CONTRACT_HISTORY_PRO_24.csv",
                index=False
            )
        summary_df.to_csv(
            OUTPUT_PATH / f"{symbol}_DUAL_MODEL_SCORECARD_PRO_24.csv",
            index=False
        )
        market_df.to_csv(
            OUTPUT_PATH / f"{symbol}_MARKET_INTELLIGENCE_PRO_24.csv",
            index=False
        )
        pair.to_csv(
            OUTPUT_PATH / f"{symbol}_FIXED_PAIR_PRO_24.csv",
            index=False
        )
        print(f"\nCSV exports saved to: {OUTPUT_PATH}")

    if EXPORT_EXCEL:
        excel_file = OUTPUT_PATH / f"{symbol}_FUTURES_INTELLIGENCE_PRO_24.xlsx"
        try:
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Dual_Model_Scorecard", index=False)
                market_df.to_excel(writer, sheet_name="Market_Intelligence", index=False)
                history_df.to_excel(writer, sheet_name="Contract_History", index=False)
                pair.to_excel(writer, sheet_name="Fixed_Near_Next_Pair", index=False)

                for ws in writer.sheets.values():
                    ws.freeze_panes = "A2"
                    ws.auto_filter.ref = ws.dimensions
                    for cells in ws.columns:
                        max_len = max(
                            (len(str(c.value)) for c in cells if c.value is not None),
                            default=10
                        )
                        ws.column_dimensions[cells[0].column_letter].width = min(max_len + 2, 30)

            print(f"Excel workbook saved: {excel_file}")
        except ImportError:
            print(
                "Excel export skipped: openpyxl is missing. "
                "Install with: python -m pip install openpyxl"
            )
        except Exception as exc:
            print(f"Excel export error: {exc}")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis(symbol):
    df = load_data(symbol)
    if df.empty:
        return

    active = get_active_contracts(df)
    if not active:
        print("No active contracts found.")
        return

    histories = {}
    summaries = []

    for contract in active:
        expiry_type = contract["EXPIRY_TYPE"]
        expiry_date = contract["EXP_DATE"]

        # Track the same actual expiry contract across role transitions.
        contract_history = df[df["EXP_DATE"] == expiry_date].copy()
        contract_history = add_contract_metrics(contract_history)
        contract_history["CONTRACT_ID"] = (
            symbol + "_" + expiry_date.strftime("%Y%m%d")
        )
        contract_history["CURRENT_EXPIRY_TYPE"] = expiry_type

        histories[expiry_type] = contract_history
        summary = build_contract_summary(contract_history, expiry_type)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("No valid contract summaries generated.")
        return

    market = aggregate_market_scores(summaries)

    near = next((c for c in active if c["EXPIRY_TYPE"] == "NEAR"), None)
    nxt = next((c for c in active if c["EXPIRY_TYPE"] == "NEXT"), None)

    if near is not None and nxt is not None:
        pair = build_fixed_pair(df, near["EXP_DATE"], nxt["EXP_DATE"])
    else:
        pair = pd.DataFrame()

    print_reports(summaries, market)
    export_results(symbol, histories, summaries, pair, market)
    create_dashboard(symbol, histories, summaries, pair, market)

    print("\nANALYSIS COMPLETED.")


def main():
    print("\n" + "=" * 82)
    print(f"       MARKETFORGE FUTURES OI ANALYZER PRO {VERSION}")
    print("       DUAL-MODEL POSITIONING INTELLIGENCE")
    print("=" * 82)
    print(f"MarketForge Root: {MARKETFORGE_ROOT}")
    print(f"Data Path: {DATA_ROOT}")
    print("Sources: FUTSTK + FUTIDX")
    print("Contract identity: SYMBOL + EXP_DATE")
    print("Signals: 1D | 3D | 5D")
    print("Dual model: Fresh positioning + Position closure")
    print("Expiry weights: NEAR 50% | NEXT 30% | FAR 20%")
    print("=" * 82)

    if not FUTSTK_PATH.is_dir() or not FUTIDX_PATH.is_dir():
        print("\nERROR: Futures data folders were not found.")
        print("Check MARKETFORGE_ROOT or edit the environment variable.")
        return

    while True:
        symbol = input("\nEnter Stock/Index (or EXIT): ").strip().upper()
        if symbol == "EXIT":
            print("\nMarketForge Analyzer closed.")
            break
        if not symbol:
            continue
        try:
            run_analysis(symbol)
        except Exception as exc:
            print(f"\nANALYSIS ERROR for {symbol}: {exc}")
            print("Review the input CSV and the error above.")


if __name__ == "__main__":
    main()
