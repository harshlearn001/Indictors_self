import pandas as pd
import os

# ===============================
# PATHS
# ===============================
price_stock_path = r"H:\MarketForge\data\master\Equity_stock_master"
futstk_path = r"H:\MarketForge\data\master\Futures_master\FUTSTK"

# ===============================
# SETTINGS
# ===============================
window = 50
lookback = 20
swing_lookback = 10

capital = 1_000_000
risk_per_trade = 0.01

# ===============================
# SMART DATE PARSER
# ===============================
def smart_date_parse(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
    else:
        return pd.to_datetime(series, errors="coerce")

# ===============================
# SIGNAL CLASSIFIER
# ===============================
def classify(price_change, oi_change):
    if price_change > 0 and oi_change > 0:
        return "LONG_BUILDUP"
    elif price_change < 0 and oi_change > 0:
        return "SHORT_BUILDUP"
    elif price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    elif price_change < 0 and oi_change < 0:
        return "LONG_UNWINDING"
    else:
        return "NEUTRAL"

# ===============================
# SCAN
# ===============================
results = []
files = [f for f in os.listdir(price_stock_path) if f.endswith(".csv")]

print(f"🔍 Scanning {len(files)} stocks...\n")

for file in files:
    symbol = file.replace(".csv", "")

    try:
        price_file = os.path.join(price_stock_path, file)
        oi_file = os.path.join(futstk_path, file)

        if not os.path.exists(oi_file):
            continue

        # ===== LOAD PRICE =====
        price_df = pd.read_csv(price_file)
        price_df.columns = price_df.columns.str.upper()

        date_col = next((c for c in ["DATE","TIMESTAMP","TRADE_DATE"] if c in price_df.columns), None)
        if date_col is None:
            continue

        price_df[date_col] = smart_date_parse(price_df[date_col])
        price_df = price_df.dropna(subset=[date_col]).sort_values(date_col)

        # ===== LIQUIDITY FILTER =====
        if "TRD_QTY" in price_df.columns:
            if price_df["TRD_QTY"].rolling(20).mean().iloc[-1] < 200000:
                continue

        price_df = price_df[[date_col, "CLOSE"]].rename(columns={date_col: "DATE"})

        # ===== LOAD OI =====
        oi_df = pd.read_csv(oi_file)
        oi_df.columns = oi_df.columns.str.upper()

        oi_col = next((c for c in ["OI","OPEN_INT","OPENINTEREST"] if c in oi_df.columns), None)
        date_col_oi = next((c for c in ["DATE","TIMESTAMP","TRADE_DATE"] if c in oi_df.columns), None)

        if oi_col is None or date_col_oi is None:
            continue

        oi_df[date_col_oi] = smart_date_parse(oi_df[date_col_oi])
        oi_df = oi_df.dropna(subset=[date_col_oi])

        oi_df = oi_df.groupby(date_col_oi)[oi_col].sum().reset_index()
        oi_df = oi_df.rename(columns={date_col_oi: "DATE", oi_col: "OI"})

        # ===== MERGE =====
        df = pd.merge(price_df, oi_df, on="DATE", how="inner")

        if len(df) < window:
            continue

        df = df.sort_values("DATE")

        # ===== SMOOTH =====
        df["CLOSE_SMOOTH"] = df["CLOSE"].rolling(5).mean()
        df["OI_SMOOTH"] = df["OI"].rolling(5).mean()

        # ===== PERCENTILES =====
        df["PRICE_PCT"] = df["CLOSE_SMOOTH"].rolling(window).rank(pct=True) * 100
        df["OI_PCT"] = df["OI_SMOOTH"].rolling(window).rank(pct=True) * 100

        df = df.dropna()
        if len(df) < 2:
            continue

        latest = df.iloc[-1]

        price_change = df["CLOSE"].iloc[-1] - df["CLOSE"].iloc[-2]
        oi_change = df["OI"].iloc[-1] - df["OI"].iloc[-2]

        signal = classify(price_change, oi_change)

        # ===== ENTRY =====
        recent_high = df["CLOSE"].rolling(lookback).max().iloc[-1]
        recent_low = df["CLOSE"].rolling(lookback).min().iloc[-1]
        current_price = df["CLOSE"].iloc[-1]

        long_entry = current_price >= 0.98 * recent_high
        short_entry = current_price <= 1.02 * recent_low

        # ===== RISK =====
        swing_low = df["CLOSE"].rolling(swing_lookback).min().iloc[-1]
        swing_high = df["CLOSE"].rolling(swing_lookback).max().iloc[-1]

        sl = None
        target = None

        if long_entry:
            sl = swing_low
            risk = current_price - sl
            target = current_price + (2 * risk)

        elif short_entry:
            sl = swing_high
            risk = sl - current_price
            target = current_price - (2 * risk)

        # ===== ATR FILTER =====
        atr = df["CLOSE"].diff().abs().rolling(14).mean().iloc[-1]

        if sl:
            risk_abs = abs(current_price - sl)
            if not (0.5 * atr <= risk_abs <= 3 * atr):
                continue

        # ===== POSITION SIZE =====
        qty = None
        if sl and current_price != sl:
            per_share_risk = abs(current_price - sl)
            capital_risk = capital * risk_per_trade
            qty = int(capital_risk // per_share_risk)

        # ===== STORE =====
        results.append({
            "Symbol": symbol,
            "Price": round(current_price, 2),
            "SL": round(sl, 2) if sl else None,
            "Target": round(target, 2) if target else None,
            "Qty": qty,
            "Signal": signal,
            "Strength": round(latest["OI_PCT"]*0.6 + latest["PRICE_PCT"]*0.4, 2),
            "Long_Entry": long_entry,
            "Short_Entry": short_entry
        })

    except:
        continue

# ===============================
# DATAFRAME
# ===============================
df_result = pd.DataFrame(results)

if df_result.empty:
    print("❌ No trades found")
    exit()

# ===============================
# RISK %
# ===============================
df_result["RiskPct"] = abs(df_result["Price"] - df_result["SL"]) / df_result["Price"] * 100

# ===============================
# FINAL FILTER
# ===============================
longs = df_result[
    (df_result["Signal"] == "LONG_BUILDUP") &
    (df_result["Long_Entry"] == True) &
    (df_result["Strength"] >= 60) &
    (df_result["RiskPct"] <= 3)
].sort_values(by="Strength", ascending=False)

shorts = df_result[
    (df_result["Signal"] == "SHORT_BUILDUP") &
    (df_result["Short_Entry"] == True) &
    (df_result["Strength"] >= 60) &
    (df_result["RiskPct"] <= 3)
].sort_values(by="Strength", ascending=False)

# ===============================
# BEST PICKS
# ===============================
best_long = longs.head(1)
best_short = shorts.head(1)

# ===============================
# OUTPUT
# ===============================
print("\n🎯 BEST LONG TRADE")
if not best_long.empty:
    print(best_long[["Symbol","Price","SL","Target","Qty","Strength","RiskPct"]])
else:
    print("No high-quality long trade")

print("\n🎯 BEST SHORT TRADE")
if not best_short.empty:
    print(best_short[["Symbol","Price","SL","Target","Qty","Strength","RiskPct"]])
else:
    print("No high-quality short trade")