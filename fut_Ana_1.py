"""
MARKETFORGE FUTURES INTELLIGENCE SCANNER 3.1
Integrates Pro 2.5 stock and index futures analysis, plus index-options expiry intelligence.

Core design:
- Analyze NEAR, NEXT and FAR separately using SYMBOL + EXP_DATE contract identity.
- Pro 2.5 expiry weights: NEAR 50%, NEXT 30%, FAR 20%.
- Pro 2.5 timeframe weights: 1D 50%, 3D 30%, 5D 20%.
- Keep fresh positioning separate from position closure:
    FRESH_BULLISH = LONG_BUILDUP
    FRESH_BEARISH = SHORT_BUILDUP
    BULLISH_CLOSURE = SHORT_COVERING
    BEARISH_CLOSURE = LONG_UNWINDING
- Calculate expiry agreement and coverage; confidence is NOT probability of profit.
- Use equity prices for stock risk; NEAR index-futures OHLC for index risk.
- FUTIDX stays monthly; OPTIDX handles NIFTY weekly+monthly and other index monthly expiries.
- Show discovery rankings separately from risk-qualified candidates.
- OI composition is descriptive only, not verified exchange rollover volume.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
import math
import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MARKETFORGE_ROOT = Path(r"H:\MarketForge")
DATA_ROOT = MARKETFORGE_ROOT / "data" / "master" / "Futures_master_three_expiries"
FUTSTK_PATH = DATA_ROOT / "FUTSTK"
FUTIDX_PATH = DATA_ROOT / "FUTIDX"
INDEX_OPTIONS_PATH = MARKETFORGE_ROOT / "data" / "master" / "option_master" / "INDICES"
INDEX_PRICE_PATH = MARKETFORGE_ROOT / "data" / "master" / "Indices_master"
PRICE_PATH = MARKETFORGE_ROOT / "data" / "master" / "Equity_stock_master"
OUTPUT_PATH = Path(r"H:\Indictors_self\output")


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "3.1"
EXPIRIES = ("NEAR", "NEXT", "FAR")
EXPIRY_WEIGHTS = {"NEAR": 0.50, "NEXT": 0.30, "FAR": 0.20}
TIMEFRAME_WEIGHTS = {1: 0.50, 3: 0.30, 5: 0.20}

MAX_DATA_AGE_DAYS = 5
MIN_CONTRACT_SESSIONS = 12
MIN_PRICE_SESSIONS = 60
LOOKBACK_SESSIONS = 25

ENTRY_LOOKBACK = 20
ENTRY_PROXIMITY_PCT = 2.0
SWING_LOOKBACK = 10
ATR_PERIOD = 14

CAPITAL = 1_000_000
RISK_PER_TRADE = 0.01
MAX_RISK_PCT = 3.0
MIN_STOP_ATR = 0.5
MAX_STOP_ATR = 3.0
REWARD_RISK = 2.0

DISCOVERY_TOP_N = 10
CONFIRMED_TOP_N = 5
MIN_DIRECTION_SCORE = 50.0
MIN_EXPIRY_AGREEMENT_PCT = 60.0

DATE_ALIASES = ("TRADE_DATE", "DATE", "TIMESTAMP")
EXPIRY_ALIASES = ("EXP_DATE", "EXPIRY", "EXPIRY_DT", "EXPIRY_DATE")
CLOSE_ALIASES = ("CLOSE_PRICE", "CLOSE", "LAST_PRICE")
OI_ALIASES = ("OPEN_INT", "OI", "OPEN_INTEREST", "OPENINTEREST")
PRICE_DATE_ALIASES = ("DATE", "TRADE_DATE", "TIMESTAMP")
PRICE_CLOSE_ALIASES = ("CLOSE", "CLOSE_PRICE", "LAST_PRICE")
HIGH_ALIASES = ("HIGH", "HI_PRICE")
LOW_ALIASES = ("LOW", "LO_PRICE")
QTY_ALIASES = ("TRD_QTY", "TOTTRDQTY", "TOTAL_TRADED_QTY", "VOLUME")


# ============================================================
# HELPERS
# ============================================================

def find_col(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lookup = {str(c).strip().upper(): c for c in df.columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    compact = text.str.fullmatch(r"\d{8}", na=False)
    iso = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    other = ~(compact | iso)
    if compact.any():
        out.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if iso.any():
        out.loc[iso] = pd.to_datetime(text.loc[iso], format="%Y-%m-%d", errors="coerce")
    if other.any():
        out.loc[other] = pd.to_datetime(text.loc[other], dayfirst=True, errors="coerce")
    return out


def weighted_average(values, weights):
    pairs = [(float(v), float(w)) for v, w in zip(values, weights)
             if pd.notna(v) and w > 0]
    if not pairs:
        return np.nan
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total


def classify(price_pct, oi_pct):
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


def file_map(folder: Path) -> dict[str, Path]:
    return {p.stem.strip().upper(): p for p in folder.glob("*.csv")}


# ============================================================
# LOADERS
# ============================================================

def load_futures_symbol(path: Path, symbol: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().upper() for c in raw.columns]

    date_col = find_col(raw, DATE_ALIASES)
    expiry_col = find_col(raw, EXPIRY_ALIASES)
    close_col = find_col(raw, CLOSE_ALIASES)
    oi_col = find_col(raw, OI_ALIASES)

    required = {
        "TRADE_DATE": date_col,
        "EXP_DATE": expiry_col,
        "CLOSE_PRICE": close_col,
        "OPEN_INT": oi_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"{symbol}: futures file missing required fields {missing}; columns={list(raw.columns)}")

    symbol_col = find_col(raw, ("SYMBOL",))
    expiry_type_col = find_col(raw, ("EXPIRY_TYPE",))
    if symbol_col is None or expiry_type_col is None:
        raise ValueError(f"{symbol}: FUTSTK file must contain SYMBOL and EXPIRY_TYPE")

    df = raw.copy()
    df["TRADE_DATE"] = parse_dates(df[date_col])
    df["EXP_DATE"] = parse_dates(df[expiry_col])
    df["SYMBOL"] = df[symbol_col].astype("string").str.strip().str.upper()
    df["EXPIRY_TYPE"] = df[expiry_type_col].astype("string").str.strip().str.upper()
    df["CLOSE_PRICE"] = pd.to_numeric(df[close_col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    df["OPEN_INT"] = pd.to_numeric(df[oi_col].astype("string").str.replace(",", "", regex=False), errors="coerce")

    for optional in ("HI_PRICE", "LO_PRICE", "TRD_QTY", "TRD_VAL", "NO_OF_CONT", "NO_OF_TRADE"):
        col = find_col(df, (optional,))
        if col is not None:
            df[optional] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["SYMBOL"] == symbol].copy()
    df = df[df["EXPIRY_TYPE"].isin(EXPIRIES)]
    df.dropna(subset=["TRADE_DATE", "EXP_DATE", "CLOSE_PRICE", "OPEN_INT"], inplace=True)
    df = df[(df["CLOSE_PRICE"] > 0) & (df["OPEN_INT"] >= 0)]
    df.sort_values(["TRADE_DATE", "EXP_DATE"], inplace=True)

    # Exact Pro 2.5 contract identity checks.
    duplicate_contracts = int(df.duplicated(["TRADE_DATE", "SYMBOL", "EXP_DATE"]).sum())
    duplicate_buckets = int(df.duplicated(["TRADE_DATE", "EXPIRY_TYPE"]).sum())
    if duplicate_contracts:
        raise ValueError(f"{symbol}: {duplicate_contracts} duplicate TRADE_DATE+SYMBOL+EXP_DATE rows")
    if duplicate_buckets:
        raise ValueError(f"{symbol}: {duplicate_buckets} duplicate TRADE_DATE+EXPIRY_TYPE rows")

    if df.empty:
        raise ValueError(f"{symbol}: no valid futures rows after cleaning")

    latest_date = df["TRADE_DATE"].max()
    age = (pd.Timestamp(date.today()) - latest_date.normalize()).days
    if age < 0 or age > MAX_DATA_AGE_DAYS:
        raise ValueError(f"{symbol}: latest futures date {latest_date.date()} is {age} days old")

    latest = df[df["TRADE_DATE"] == latest_date]
    active_types = set(latest["EXPIRY_TYPE"])
    missing_expiries = [e for e in EXPIRIES if e not in active_types]
    if missing_expiries:
        raise ValueError(f"{symbol}: missing active expiry buckets {missing_expiries}")

    latest_expiries = (
        latest.drop_duplicates("EXPIRY_TYPE")
        .set_index("EXPIRY_TYPE")["EXP_DATE"].to_dict()
    )
    ordered = [latest_expiries[e] for e in EXPIRIES]
    if any(ordered[i] >= ordered[i + 1] for i in range(2)):
        raise ValueError(f"{symbol}: NEAR/NEXT/FAR expiry dates not chronological")
    if any(x.normalize() <= latest_date.normalize() for x in ordered):
        raise ValueError(f"{symbol}: active expiry date is not after latest trade date")

    return df.reset_index(drop=True)


def load_equity_prices(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    dcol = find_col(raw, PRICE_DATE_ALIASES)
    ccol = find_col(raw, PRICE_CLOSE_ALIASES)
    hcol = find_col(raw, HIGH_ALIASES)
    lcol = find_col(raw, LOW_ALIASES)
    qcol = find_col(raw, QTY_ALIASES)
    if dcol is None or ccol is None:
        raise ValueError(f"Equity file missing date/close columns: {list(raw.columns)}")

    out = pd.DataFrame()
    out["DATE"] = parse_dates(raw[dcol])
    out["CLOSE"] = pd.to_numeric(raw[ccol], errors="coerce")
    if hcol is not None:
        out["HIGH"] = pd.to_numeric(raw[hcol], errors="coerce")
    if lcol is not None:
        out["LOW"] = pd.to_numeric(raw[lcol], errors="coerce")
    if qcol is not None:
        out["TRD_QTY"] = pd.to_numeric(raw[qcol], errors="coerce")

    out.dropna(subset=["DATE", "CLOSE"], inplace=True)
    out = out[out["CLOSE"] > 0].sort_values("DATE")
    out.drop_duplicates("DATE", keep="last", inplace=True)
    return out.reset_index(drop=True)


# ============================================================
# PRO 2.5 CONTRACT METRICS AND SCORING
# ============================================================

def add_pro25_metrics(history: pd.DataFrame) -> pd.DataFrame:
    h = history.sort_values("TRADE_DATE").reset_index(drop=True).copy()
    h["OI_CHANGE"] = h["OPEN_INT"].diff()
    h["PRICE_CHANGE"] = h["CLOSE_PRICE"].diff()

    for period in (1, 3, 5, 10):
        h[f"OI_{period}D_PCT"] = h["OPEN_INT"].pct_change(period, fill_method=None) * 100
        h[f"PRICE_{period}D_PCT"] = h["CLOSE_PRICE"].pct_change(period, fill_method=None) * 100

    for period in (1, 3, 5):
        h[f"SIGNAL_{period}D"] = [
            classify(p, o)
            for p, o in zip(h[f"PRICE_{period}D_PCT"], h[f"OI_{period}D_PCT"])
        ]

    h["OI_MOMENTUM_CHANGE_3D_PP"] = h["OI_3D_PCT"] - h["OI_3D_PCT"].shift(3)
    h["OI_MOMENTUM_CHANGE_5D_PP"] = h["OI_5D_PCT"] - h["OI_5D_PCT"].shift(5)

    def expansion(row):
        v = weighted_average(
            [row["OI_1D_PCT"], row["OI_3D_PCT"], row["OI_5D_PCT"]],
            [0.50, 0.30, 0.20],
        )
        if pd.isna(v):
            return np.nan
        return float(np.clip(50 + 50 * math.tanh(v / 25.0), 0, 100))

    h["OI_EXPANSION_SCORE"] = h.apply(expansion, axis=1)
    h["PREVIOUS_EXPIRY_TYPE"] = h["EXPIRY_TYPE"].shift(1)
    h["ROLE_TRANSITION"] = h["PREVIOUS_EXPIRY_TYPE"].notna() & h["EXPIRY_TYPE"].ne(h["PREVIOUS_EXPIRY_TYPE"])
    return h


SIGNAL_COMPONENT = {
    "LONG_BUILDUP": "FRESH_BULLISH",
    "SHORT_BUILDUP": "FRESH_BEARISH",
    "SHORT_COVERING": "BULLISH_CLOSURE",
    "LONG_UNWINDING": "BEARISH_CLOSURE",
    "NEUTRAL": None,
}


def score_pro25_contract(history: pd.DataFrame, expiry_type: str) -> dict:
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
        if pd.isna(latest[f"OI_{period}D_PCT"]) or pd.isna(latest[f"PRICE_{period}D_PCT"]):
            continue
        valid_weights.append(weight)
        component = SIGNAL_COMPONENT[signal]
        if component:
            components[component].append(weight)

    denominator = sum(valid_weights)
    scores = {
        name: (round(100 * sum(weights) / denominator, 2) if denominator else np.nan)
        for name, weights in components.items()
    }

    fb, fr = scores["FRESH_BULLISH"], scores["FRESH_BEARISH"]
    bc, rc = scores["BULLISH_CLOSURE"], scores["BEARISH_CLOSURE"]
    if any(pd.isna(x) for x in (fb, fr, bc, rc)):
        net = np.nan
    else:
        net = round(float(np.clip(50 + ((fb + bc) - (fr + rc)) / 2, 0, 100)), 2)

    return {
        "Expiry": expiry_type,
        "ExpiryDate": latest["EXP_DATE"].date().isoformat(),
        "FuturesPrice": float(latest["CLOSE_PRICE"]),
        "OI": float(latest["OPEN_INT"]),
        "OI_1D_Pct": float(latest["OI_1D_PCT"]) if pd.notna(latest["OI_1D_PCT"]) else np.nan,
        "OI_3D_Pct": float(latest["OI_3D_PCT"]) if pd.notna(latest["OI_3D_PCT"]) else np.nan,
        "OI_5D_Pct": float(latest["OI_5D_PCT"]) if pd.notna(latest["OI_5D_PCT"]) else np.nan,
        "OI_10D_Pct": float(latest["OI_10D_PCT"]) if pd.notna(latest["OI_10D_PCT"]) else np.nan,
        "Price_1D_Pct": float(latest["PRICE_1D_PCT"]) if pd.notna(latest["PRICE_1D_PCT"]) else np.nan,
        "Price_3D_Pct": float(latest["PRICE_3D_PCT"]) if pd.notna(latest["PRICE_3D_PCT"]) else np.nan,
        "Price_5D_Pct": float(latest["PRICE_5D_PCT"]) if pd.notna(latest["PRICE_5D_PCT"]) else np.nan,
        "Signal_1D": latest["SIGNAL_1D"],
        "Signal_3D": latest["SIGNAL_3D"],
        "Signal_5D": latest["SIGNAL_5D"],
        **scores,
        "NET_POSITIONING_SCORE": net,
        "OI_EXPANSION_SCORE": float(latest["OI_EXPANSION_SCORE"]) if pd.notna(latest["OI_EXPANSION_SCORE"]) else np.nan,
        "OI_MOMENTUM_CHANGE_3D_PP": float(latest["OI_MOMENTUM_CHANGE_3D_PP"]) if pd.notna(latest["OI_MOMENTUM_CHANGE_3D_PP"]) else np.nan,
        "OI_MOMENTUM_CHANGE_5D_PP": float(latest["OI_MOMENTUM_CHANGE_5D_PP"]) if pd.notna(latest["OI_MOMENTUM_CHANGE_5D_PP"]) else np.nan,
        "Sessions": len(history),
        "RoleTransitions": int(history["ROLE_TRANSITION"].sum()),
        "ValidTimeframeWeight": denominator,
    }


def aggregate_pro25(expiry_summaries: list[dict]) -> dict:
    by_expiry = {x["Expiry"]: x for x in expiry_summaries}
    valid = [by_expiry[e] for e in EXPIRIES if e in by_expiry]
    if len(valid) != 3:
        raise ValueError("All three expiries are required for aggregate scoring")

    fields = [
        "FRESH_BULLISH", "FRESH_BEARISH", "BULLISH_CLOSURE",
        "BEARISH_CLOSURE", "NET_POSITIONING_SCORE", "OI_EXPANSION_SCORE",
        "OI_MOMENTUM_CHANGE_3D_PP", "OI_MOMENTUM_CHANGE_5D_PP",
    ]
    agg = {}
    for field in fields:
        vals, weights = [], []
        for item in valid:
            value = item.get(field, np.nan)
            if pd.notna(value):
                vals.append(value)
                weights.append(EXPIRY_WEIGHTS[item["Expiry"]])
        agg[field] = round(weighted_average(vals, weights), 2) if vals else np.nan

    directions = []
    for item in valid:
        score = item["NET_POSITIONING_SCORE"]
        direction = "BULLISH" if score >= 60 else "BEARISH" if score <= 40 else "MIXED"
        directions.append((direction, EXPIRY_WEIGHTS[item["Expiry"]]))

    overall = "BULLISH" if agg["NET_POSITIONING_SCORE"] >= 60 else (
        "BEARISH" if agg["NET_POSITIONING_SCORE"] <= 40 else "MIXED"
    )
    agreement_weight = sum(w for d, w in directions if d == overall)
    total_weight = sum(w for _, w in directions)
    agreement = 100 * agreement_weight / total_weight if total_weight else 0
    strength = min(1.0, abs(agg["NET_POSITIONING_SCORE"] - 50) / 50)
    coverage = len(valid) / 3
    confidence = round(100 * (0.50 * agreement / 100 + 0.30 * strength + 0.20 * coverage), 1)

    agg["OVERALL_POSITIONING"] = overall
    agg["EXPIRY_AGREEMENT_PCT"] = round(agreement, 1)
    agg["VALID_EXPIRIES"] = len(valid)
    agg["DATA_COVERAGE_PCT"] = round(coverage * 100, 1)
    agg["CONFIDENCE_SCORE"] = confidence
    agg["CONFIDENCE_LABEL"] = (
        "STRONG" if confidence >= 75 and agreement >= 75
        else "MODERATE" if confidence >= 55 and agreement >= 50
        else "LOW"
    )
    return agg


# ============================================================
# EQUITY ENTRY / RISK MODEL
# ============================================================

def equity_risk_context(prices: pd.DataFrame) -> dict:
    p = prices.sort_values("DATE").copy()
    p["ATR"] = np.nan

    if {"HIGH", "LOW"}.issubset(p.columns):
        prev = p["CLOSE"].shift(1)
        tr = pd.concat([
            p["HIGH"] - p["LOW"],
            (p["HIGH"] - prev).abs(),
            (p["LOW"] - prev).abs(),
        ], axis=1).max(axis=1)
        p["ATR"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
        atr_method = "TRUE_RANGE"
    else:
        p["ATR"] = p["CLOSE"].diff().abs().rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
        atr_method = "CLOSE_CHANGE_PROXY"

    p["PRIOR_HIGH"] = p["CLOSE"].shift(1).rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).max()
    p["PRIOR_LOW"] = p["CLOSE"].shift(1).rolling(ENTRY_LOOKBACK, min_periods=ENTRY_LOOKBACK).min()
    p["SWING_LOW"] = p["CLOSE"].rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).min()
    p["SWING_HIGH"] = p["CLOSE"].rolling(SWING_LOOKBACK, min_periods=SWING_LOOKBACK).max()
    p["AVG_QTY_20D"] = p["TRD_QTY"].rolling(20, min_periods=20).mean() if "TRD_QTY" in p else np.nan

    p.dropna(subset=["ATR", "PRIOR_HIGH", "PRIOR_LOW", "SWING_LOW", "SWING_HIGH"], inplace=True)
    if len(p) < 1:
        raise ValueError("Equity price indicators are not ready")

    row = p.iloc[-1]
    age = (pd.Timestamp(date.today()) - row["DATE"].normalize()).days
    if age < 0 or age > MAX_DATA_AGE_DAYS:
        raise ValueError(f"Equity price data stale/future dated: {row['DATE'].date()}, age={age}")

    return {
        "TradeDate": row["DATE"].date().isoformat(),
        "EquityPrice": float(row["CLOSE"]),
        "ATR": float(row["ATR"]),
        "ATR_Method": atr_method,
        "Prior20High": float(row["PRIOR_HIGH"]),
        "Prior20Low": float(row["PRIOR_LOW"]),
        "SwingLow": float(row["SWING_LOW"]),
        "SwingHigh": float(row["SWING_HIGH"]),
        "AvgTrdQty20D": float(row["AVG_QTY_20D"]) if pd.notna(row["AVG_QTY_20D"]) else np.nan,
        "DataAgeDays": age,
        "PriceSessions": len(prices),
    }


def direction_trade_fields(context: dict, direction: str) -> dict:
    price = context["EquityPrice"]
    atr = context["ATR"]
    if direction == "BULLISH":
        stop = context["SwingLow"]
        risk = price - stop
        entry_ref = context["Prior20High"]
        near_entry = price >= entry_ref * (1 - ENTRY_PROXIMITY_PCT / 100)
        target = price + REWARD_RISK * risk
    else:
        stop = context["SwingHigh"]
        risk = stop - price
        entry_ref = context["Prior20Low"]
        near_entry = price <= entry_ref * (1 + ENTRY_PROXIMITY_PCT / 100)
        target = price - REWARD_RISK * risk

    risk_pct = risk / price * 100 if price > 0 else np.nan
    risk_atr = risk / atr if atr > 0 else np.nan
    risk_ok = (
        risk > 0 and np.isfinite(risk_atr)
        and MIN_STOP_ATR <= risk_atr <= MAX_STOP_ATR
        and risk_pct <= MAX_RISK_PCT
    )
    qty = int((CAPITAL * RISK_PER_TRADE) // risk) if risk > 0 else 0
    return {
        "EntryReference": round(entry_ref, 2),
        "EntryProximityOK": bool(near_entry),
        "StopLoss": round(stop, 2),
        "Target_2R": round(target, 2) if risk > 0 else np.nan,
        "RiskPct": round(risk_pct, 2) if np.isfinite(risk_pct) else np.nan,
        "RiskATR": round(risk_atr, 2) if np.isfinite(risk_atr) else np.nan,
        "RiskOK": bool(risk_ok),
        "RiskQty_Shares": qty,
    }


# ============================================================
# SYMBOL SCAN
# ============================================================

def scan_symbol(symbol: str, futures_file: Path, price_file: Path):
    futures = load_futures_symbol(futures_file, symbol)
    prices = load_equity_prices(price_file)
    if len(prices) < MIN_PRICE_SESSIONS:
        raise ValueError(f"Insufficient equity price history: {len(prices)} sessions")

    context = equity_risk_context(prices)
    latest_fut_date = futures["TRADE_DATE"].max().normalize()
    if latest_fut_date.date().isoformat() != context["TradeDate"]:
        # Permit different source calendars only if same date is available in both.
        matched = prices[prices["DATE"].dt.normalize() == latest_fut_date]
        if matched.empty:
            raise ValueError(
                f"Price/futures date mismatch: futures={latest_fut_date.date()}, "
                f"equity latest={context['TradeDate']}"
            )
        # Recompute context using prices through the futures latest date.
        context = equity_risk_context(prices[prices["DATE"] <= latest_fut_date].copy())

    latest_rows = futures[futures["TRADE_DATE"].dt.normalize() == latest_fut_date]
    active_contracts = (
        latest_rows.sort_values("EXP_DATE")
        .drop_duplicates("EXPIRY_TYPE")
        .set_index("EXPIRY_TYPE")["EXP_DATE"].to_dict()
    )

    expiry_summaries = []
    for expiry_type in EXPIRIES:
        expiry_date = active_contracts[expiry_type]
        # Track the same actual expiry contract across NEAR/NEXT/FAR role changes.
        contract_history = futures[futures["EXP_DATE"] == expiry_date].copy()
        contract_history = add_pro25_metrics(contract_history)
        if len(contract_history) < MIN_CONTRACT_SESSIONS:
            raise ValueError(
                f"{expiry_type} contract has only {len(contract_history)} sessions; "
                f"minimum is {MIN_CONTRACT_SESSIONS}"
            )
        expiry_summaries.append(score_pro25_contract(contract_history, expiry_type))

    aggregate = aggregate_pro25(expiry_summaries)
    bullish_direction_score = round(
        0.60 * aggregate["FRESH_BULLISH"] +
        0.40 * aggregate["NET_POSITIONING_SCORE"], 2
    )
    bearish_direction_score = round(
        0.60 * aggregate["FRESH_BEARISH"] +
        0.40 * (100 - aggregate["NET_POSITIONING_SCORE"]), 2
    )

    # Directional confirmation uses Pro 2.5 fresh-positioning scores and
    # expiry agreement. Closure is reported separately, never relabelled fresh buildup.
    bull = (
        bullish_direction_score >= MIN_DIRECTION_SCORE
        and aggregate["OVERALL_POSITIONING"] == "BULLISH"
    )
    bear = (
        bearish_direction_score >= MIN_DIRECTION_SCORE
        and aggregate["OVERALL_POSITIONING"] == "BEARISH"
    )
    if bull and bear:
        direction = "MIXED"
    elif bull:
        direction = "BULLISH"
    elif bear:
        direction = "BEARISH"
    else:
        direction = "NONE"

    base = {
        "Symbol": symbol,
        **context,
        **aggregate,
        "BullDirectionScore": bullish_direction_score,
        "BearDirectionScore": bearish_direction_score,
    }

    per_expiry = []
    for item in expiry_summaries:
        per_expiry.append({"Symbol": symbol, **item})

    candidates = []
    if direction in ("BULLISH", "BEARISH"):
        score = bullish_direction_score if direction == "BULLISH" else bearish_direction_score
        risk_fields = direction_trade_fields(context, direction)
        closure_score = aggregate["BULLISH_CLOSURE"] if direction == "BULLISH" else aggregate["BEARISH_CLOSURE"]
        opposite_fresh = aggregate["FRESH_BEARISH"] if direction == "BULLISH" else aggregate["FRESH_BULLISH"]

        # Penalize contradictory fresh positioning and report closure independently.
        discovery_score = round(
            max(0.0, score - 0.20 * opposite_fresh), 2
        )
        candidate = {
            **base,
            "Direction": direction,
            "Strength": discovery_score,
            "FreshDirectionalScore": score,
            "OppositeFreshScore": opposite_fresh,
            "DirectionalClosureScore": closure_score,
            **risk_fields,
        }
        candidates.append(candidate)

    diagnostics = {
        **base,
        "Direction": direction,
        "BullishDiscovery": bull,
        "BearishDiscovery": bear,
        "PerExpiry": " | ".join(
            f"{x['Expiry']}:{x['NET_POSITIONING_SCORE']:.1f}/{x['Signal_1D']}/{x['Signal_3D']}/{x['Signal_5D']}"
            for x in expiry_summaries
        ),
    }
    return candidates, per_expiry, diagnostics


# ============================================================
# INDEX FUTURES + INDEX OPTIONS EXPIRY MODULES
# ============================================================

def load_index_price_for_risk(futures: pd.DataFrame, near_expiry: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build a continuous daily NEAR-futures price series for index risk context.

    OI scoring still tracks each fixed expiry separately. The continuous NEAR
    series is used only for technical entry/ATR context across monthly rolls.
    """
    h = futures[futures["EXPIRY_TYPE"] == "NEAR"].copy().sort_values("TRADE_DATE")
    h = h.drop_duplicates("TRADE_DATE", keep="last")
    out = pd.DataFrame({"DATE": h["TRADE_DATE"], "CLOSE": h["CLOSE_PRICE"]})
    if "HI_PRICE" in h.columns:
        out["HIGH"] = pd.to_numeric(h["HI_PRICE"], errors="coerce")
    if "LO_PRICE" in h.columns:
        out["LOW"] = pd.to_numeric(h["LO_PRICE"], errors="coerce")
    if "TRD_QTY" in h.columns:
        out["TRD_QTY"] = pd.to_numeric(h["TRD_QTY"], errors="coerce")
    out.dropna(subset=["DATE", "CLOSE"], inplace=True)
    out.drop_duplicates("DATE", keep="last", inplace=True)
    return out.reset_index(drop=True)


def scan_index_future(symbol: str, futures_file: Path):
    """Pro 2.5 analysis for FUTIDX: all three contracts are monthly futures."""
    futures = load_futures_symbol(futures_file, symbol)
    latest_date = futures["TRADE_DATE"].max().normalize()
    latest_rows = futures[futures["TRADE_DATE"].dt.normalize() == latest_date]
    active = (latest_rows.sort_values("EXP_DATE").drop_duplicates("EXPIRY_TYPE")
              .set_index("EXPIRY_TYPE")["EXP_DATE"].to_dict())
    near_expiry = active["NEAR"]
    prices = load_index_price_for_risk(futures, near_expiry)
    if len(prices) < MIN_PRICE_SESSIONS:
        raise ValueError(f"{symbol}: insufficient NEAR index-futures price history ({len(prices)} sessions)")
    context = equity_risk_context(prices)
    if context["TradeDate"] != latest_date.date().isoformat():
        prices = prices[prices["DATE"] <= latest_date]
        context = equity_risk_context(prices)

    summaries = []
    for expiry_type in EXPIRIES:
        expiry_date = active[expiry_type]
        h = add_pro25_metrics(futures[futures["EXP_DATE"] == expiry_date].copy())
        if len(h) < MIN_CONTRACT_SESSIONS:
            raise ValueError(f"{symbol} {expiry_type} monthly contract has only {len(h)} sessions; minimum is {MIN_CONTRACT_SESSIONS}")
        summaries.append(score_pro25_contract(h, expiry_type))

    aggregate = aggregate_pro25(summaries)
    bull_score = round(0.60 * aggregate["FRESH_BULLISH"] + 0.40 * aggregate["NET_POSITIONING_SCORE"], 2)
    bear_score = round(0.60 * aggregate["FRESH_BEARISH"] + 0.40 * (100 - aggregate["NET_POSITIONING_SCORE"]), 2)
    bull = bull_score >= MIN_DIRECTION_SCORE and aggregate["OVERALL_POSITIONING"] == "BULLISH"
    bear = bear_score >= MIN_DIRECTION_SCORE and aggregate["OVERALL_POSITIONING"] == "BEARISH"
    direction = "BULLISH" if bull and not bear else "BEARISH" if bear and not bull else "NONE"
    base = {"Market": "INDEX_FUTURES", "Symbol": symbol, **context, **aggregate,
            "BullDirectionScore": bull_score, "BearDirectionScore": bear_score,
            "FuturesExpiryCycle": "MONTHLY_NEAR_NEXT_FAR"}
    per_expiry = [{"Market": "INDEX_FUTURES", "Symbol": symbol, **x} for x in summaries]
    candidates = []
    if direction in ("BULLISH", "BEARISH"):
        score = bull_score if direction == "BULLISH" else bear_score
        opposite = aggregate["FRESH_BEARISH"] if direction == "BULLISH" else aggregate["FRESH_BULLISH"]
        closure = aggregate["BULLISH_CLOSURE"] if direction == "BULLISH" else aggregate["BEARISH_CLOSURE"]
        candidates.append({**base, "Direction": direction,
            "Strength": round(max(0.0, score - 0.20 * opposite), 2),
            "FreshDirectionalScore": score, "OppositeFreshScore": opposite,
            "DirectionalClosureScore": closure,
            **direction_trade_fields(context, direction)})
    diag = {**base, "Direction": direction,
            "PerExpiry": " | ".join(f"{x['Expiry']}:{x['NET_POSITIONING_SCORE']:.1f}/{x['Signal_1D']}/{x['Signal_3D']}/{x['Signal_5D']}" for x in summaries)}
    return candidates, per_expiry, diag


def discover_option_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted([*folder.rglob("*.parquet"), *folder.rglob("*.csv")])


def load_index_options(folder: Path) -> pd.DataFrame:
    """Load MarketForge INDICES option master (Parquet preferred; CSV also accepted)."""
    files = discover_option_files(folder)
    if not files:
        return pd.DataFrame()
    frames = []
    needed_aliases = {
        "TRADE_DATE": DATE_ALIASES,
        "EXP_DATE": EXPIRY_ALIASES,
        "SYMBOL": ("SYMBOL",),
        "OPT_TYPE": ("OPT_TYPE", "OPTION_TYPE", "OPTIONTYPE"),
        "OPEN_INT": OI_ALIASES,
        "TRD_QTY": QTY_ALIASES,
    }
    for file in files:
        try:
            raw = pd.read_parquet(file) if file.suffix.lower() == ".parquet" else pd.read_csv(file)
            raw.columns = [str(c).strip().upper() for c in raw.columns]
            cols = {k: find_col(raw, a) for k, a in needed_aliases.items()}
            if any(cols[k] is None for k in ("TRADE_DATE", "EXP_DATE", "SYMBOL", "OPT_TYPE", "OPEN_INT")):
                continue
            d = pd.DataFrame()
            d["TRADE_DATE"] = parse_dates(raw[cols["TRADE_DATE"]])
            d["EXP_DATE"] = parse_dates(raw[cols["EXP_DATE"]])
            d["SYMBOL"] = raw[cols["SYMBOL"]].astype("string").str.strip().str.upper()
            d["OPT_TYPE"] = raw[cols["OPT_TYPE"]].astype("string").str.strip().str.upper()
            d["OPEN_INT"] = pd.to_numeric(raw[cols["OPEN_INT"]], errors="coerce")
            d["TRD_QTY"] = pd.to_numeric(raw[cols["TRD_QTY"]], errors="coerce") if cols["TRD_QTY"] else np.nan
            strike_col = find_col(raw, ("STRIKE_PRICE", "STRIKE"))
            d["STRIKE_PRICE"] = pd.to_numeric(raw[strike_col], errors="coerce") if strike_col else np.nan
            d = d[d["OPT_TYPE"].isin(["CE", "PE"])]
            d.dropna(subset=["TRADE_DATE", "EXP_DATE", "SYMBOL", "OPEN_INT"], inplace=True)
            frames.append(d)
        except Exception as exc:
            print(f"[INDEX OPTIONS] Skipped {file.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data[data["OPEN_INT"] >= 0]
    # Remove duplicate rows caused by overlapping source files without collapsing strikes.
    keys = ["TRADE_DATE", "EXP_DATE", "SYMBOL", "OPT_TYPE", "STRIKE_PRICE"]
    data = data.drop_duplicates(keys, keep="last")
    return data


def classify_index_option_expiries(options: pd.DataFrame) -> pd.DataFrame:
    """NIFTY: weekly + monthly. Other indices: monthly only. Actual dates are authoritative."""
    if options.empty:
        return pd.DataFrame()
    x = options.copy()
    # Monthly expiry is the final listed expiry date in that calendar month for that index.
    expiry_calendar = (x[["SYMBOL", "EXP_DATE"]].drop_duplicates()
                       .assign(YEAR_MONTH=lambda z: z["EXP_DATE"].dt.to_period("M")))
    month_last = expiry_calendar.groupby(["SYMBOL", "YEAR_MONTH"])["EXP_DATE"].transform("max")
    expiry_calendar["IS_MONTHLY"] = expiry_calendar["EXP_DATE"].eq(month_last)
    x = x.merge(expiry_calendar[["SYMBOL", "EXP_DATE", "IS_MONTHLY"]], on=["SYMBOL", "EXP_DATE"], how="left")
    x["EXPIRY_CLASS"] = np.where(x["IS_MONTHLY"], "MONTHLY", "WEEKLY")
    # Per requested NSE universe rule: only NIFTY gets weekly contracts; all other indices monthly.
    x = x[(x["SYMBOL"].eq("NIFTY")) | (x["IS_MONTHLY"])]
    x.loc[x["SYMBOL"].ne("NIFTY"), "EXPIRY_CLASS"] = "MONTHLY"

    grouped = (x.groupby(["TRADE_DATE", "SYMBOL", "EXP_DATE", "EXPIRY_CLASS", "OPT_TYPE"], as_index=False)
               .agg(OPEN_INT=("OPEN_INT", "sum"), TRD_QTY=("TRD_QTY", "sum")))
    wide = grouped.pivot_table(index=["TRADE_DATE", "SYMBOL", "EXP_DATE", "EXPIRY_CLASS"],
                               columns="OPT_TYPE", values="OPEN_INT", aggfunc="sum", fill_value=0).reset_index()
    wide.columns.name = None
    if "CE" not in wide: wide["CE"] = 0.0
    if "PE" not in wide: wide["PE"] = 0.0
    wide.rename(columns={"CE": "CE_OI", "PE": "PE_OI"}, inplace=True)
    wide["TOTAL_OI"] = wide["CE_OI"] + wide["PE_OI"]
    wide["PCR_OI"] = np.where(wide["CE_OI"] > 0, wide["PE_OI"] / wide["CE_OI"], np.nan)
    wide.sort_values(["SYMBOL", "EXP_DATE", "TRADE_DATE"], inplace=True)
    for col in ("CE_OI", "PE_OI", "TOTAL_OI"):
        wide[f"{col}_CHG_PCT"] = wide.groupby(["SYMBOL", "EXP_DATE"])[col].pct_change(fill_method=None) * 100
    wide["PCR_CHANGE"] = wide.groupby(["SYMBOL", "EXP_DATE"])["PCR_OI"].diff()
    symbol_latest = wide.groupby("SYMBOL")["TRADE_DATE"].transform("max")
    latest = wide[wide["TRADE_DATE"] == symbol_latest].copy()
    latest["DATA_DATE"] = latest["TRADE_DATE"].dt.date.astype(str)
    return wide, latest


def run_index_options_module():
    if not INDEX_OPTIONS_PATH.is_dir():
        print(f"[INDEX OPTIONS] Not found; skipped: {INDEX_OPTIONS_PATH}")
        return pd.DataFrame(), pd.DataFrame()
    raw = load_index_options(INDEX_OPTIONS_PATH)
    if raw.empty:
        print("[INDEX OPTIONS] No compatible Parquet/CSV files found.")
        return pd.DataFrame(), pd.DataFrame()
    history, latest = classify_index_option_expiries(raw)
    return history, latest


# ============================================================
# OUTPUT / RANKING
# ============================================================

def rank(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        ["Strength", "CONFIDENCE_SCORE", "EXPIRY_AGREEMENT_PCT"],
        ascending=[False, False, False],
    ).head(n).reset_index(drop=True)


def main():
    print("=" * 100)
    print(f"MARKETFORGE FUTURES INTELLIGENCE SCANNER {VERSION}")
    print("PRO 2.5 | FUTSTK + FUTIDX MONTHLY | INDEX OPTIONS: NIFTY WEEKLY+MONTHLY / OTHERS MONTHLY")
    print("=" * 100)

    for folder in (FUTSTK_PATH, PRICE_PATH, FUTIDX_PATH):
        if not folder.is_dir():
            raise FileNotFoundError(f"Required folder not found: {folder}")

    # ---------------- STOCK FUTURES ----------------
    futures_map = file_map(FUTSTK_PATH)
    prices_map = file_map(PRICE_PATH)
    symbols = sorted(set(futures_map) & set(prices_map))
    print(f"\n[STOCK FUTURES] Files: {len(futures_map)} | Equity files: {len(prices_map)} | Matched: {len(symbols)}")
    candidates, expiry_rows, diagnostics, errors = [], [], [], []
    for i, symbol in enumerate(symbols, 1):
        print(f"[STK {i:>3}/{len(symbols)}] {symbol:<18}", end=" ", flush=True)
        try:
            found, per_expiry, diag = scan_symbol(symbol, futures_map[symbol], prices_map[symbol])
            diagnostics.append(diag); expiry_rows.extend(per_expiry); candidates.extend(found)
            if found:
                print(f"{found[0]['Direction']} score={found[0]['Strength']:.1f} risk={found[0]['RiskPct']:.2f}%")
            else:
                print(f"NO_CLEAR_DIRECTION ({diag['OVERALL_POSITIONING']})")
        except Exception as exc:
            errors.append({"Market": "STOCK_FUTURES", "Symbol": symbol, "Error": repr(exc)})
            print(f"ERROR: {exc}")

    # ---------------- INDEX FUTURES ----------------
    idx_map = file_map(FUTIDX_PATH)
    idx_candidates, idx_expiry_rows, idx_diagnostics, idx_errors = [], [], [], []
    print(f"\n[INDEX FUTURES] Files found: {len(idx_map)} | All index futures use monthly NEAR/NEXT/FAR contracts")
    for i, symbol in enumerate(sorted(idx_map), 1):
        print(f"[IDX {i:>3}/{len(idx_map)}] {symbol:<18}", end=" ", flush=True)
        try:
            found, per_expiry, diag = scan_index_future(symbol, idx_map[symbol])
            idx_diagnostics.append(diag); idx_expiry_rows.extend(per_expiry); idx_candidates.extend(found)
            if found:
                print(f"{found[0]['Direction']} score={found[0]['Strength']:.1f} risk={found[0]['RiskPct']:.2f}%")
            else:
                print(f"NO_CLEAR_DIRECTION ({diag['OVERALL_POSITIONING']})")
        except Exception as exc:
            idx_errors.append({"Market": "INDEX_FUTURES", "Symbol": symbol, "Error": repr(exc)})
            print(f"ERROR: {exc}")

    # ---------------- INDEX OPTIONS EXPIRY INTELLIGENCE ----------------
    print("\n[INDEX OPTIONS] NIFTY weekly + monthly; all other indices monthly only")
    options_history, options_latest = run_index_options_module()
    if not options_latest.empty:
        for symbol, g in options_latest.groupby("SYMBOL"):
            classes = ", ".join(sorted(g["EXPIRY_CLASS"].unique()))
            print(f"  {symbol:<16} expiry classes: {classes} | expiries in latest snapshot: {len(g)}")

    # ---------------- FINAL STOCK OUTPUTS ----------------
    # Export every directional stock candidate into exactly two Excel files.
    # Index futures/options analysis above remains unchanged; NIFTY has its own scanner.

    all_df = pd.DataFrame(candidates)

    if all_df.empty:
        bull_output = pd.DataFrame()
        bear_output = pd.DataFrame()
    else:
        sort_columns = ["Strength", "CONFIDENCE_SCORE", "EXPIRY_AGREEMENT_PCT"]
        bull_output = (
            all_df[all_df["Direction"] == "BULLISH"]
            .sort_values(sort_columns, ascending=[False, False, False])
            .reset_index(drop=True)
        )
        bear_output = (
            all_df[all_df["Direction"] == "BEARISH"]
            .sort_values(sort_columns, ascending=[False, False, False])
            .reset_index(drop=True)
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    bullish_file = OUTPUT_PATH / "Bullish.xlsx"
    bearish_file = OUTPUT_PATH / "Bearish.xlsx"

    def write_report(frame: pd.DataFrame, path: Path) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Candidates", index=False)
            ws = writer.sheets["Candidates"]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for column in ws.columns:
                width = max(
                    (len(str(cell.value)) for cell in column if cell.value is not None),
                    default=10,
                )
                ws.column_dimensions[column[0].column_letter].width = min(width + 2, 32)

    write_report(bull_output, bullish_file)
    write_report(bear_output, bearish_file)

    # Retain the original index analysis and its console summary.
    idx_all = pd.DataFrame(idx_candidates)
    bull_confirmed = pd.DataFrame()
    bear_confirmed = pd.DataFrame()

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Stock futures: directional={len(all_df)} | bullish={len(bull_output)} | bearish={len(bear_output)} | errors={len(errors)}")
    print(f"Index futures: directional={len(idx_all)} | errors={len(idx_errors)}")
    print(f"Index options latest expiry rows: {len(options_latest)}")
    print(f"Bullish report: {bullish_file}")
    print(f"Bearish report: {bearish_file}")
    print("\nEXPIRY RULES:")
    print("- FUTSTK: monthly NEAR/NEXT/FAR futures.")
    print("- FUTIDX: monthly NEAR/NEXT/FAR futures, including NIFTY.")
    print("- OPTIDX: NIFTY weekly + monthly options; other indices monthly options only.")
    print("- Option expiry classification uses actual expiry dates from the option master; holiday shifts are respected.")
    print("\nPro 2.5 confidence is positioning consistency/data coverage, NOT profit probability.")
    print("Index option PCR/OI is positioning context, not a standalone directional trade signal.")
    print("Index risk levels use the NEAR index-futures contract; lot sizing is NOT calculated.")
    print("No candidate is a live trade instruction; validate liquidity, lot size, margin and backtest.")


if __name__ == "__main__":
    main()
