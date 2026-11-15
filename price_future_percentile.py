import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# PATHS
# ===============================
price_stock_path = r"H:\MarketForge\data\master\Equity_stock_master"
futstk_path = r"H:\MarketForge\data\master\Futures_master\FUTSTK"
futidx_path = r"H:\MarketForge\data\master\Futures_master\FUTIDX"
index_path = r"H:\MarketForge\data\master\Indices_master"

# ===============================
# INPUT
# ===============================
symbol = input("Enter Symbol: ").strip().upper()

# ===============================
# DETECT PRICE SOURCE
# ===============================
price_file = None

if os.path.exists(os.path.join(price_stock_path, f"{symbol}.csv")):
    price_file = os.path.join(price_stock_path, f"{symbol}.csv")
    print("📊 Using STOCK price")

elif os.path.exists(os.path.join(index_path, f"{symbol}.csv")):
    price_file = os.path.join(index_path, f"{symbol}.csv")
    print("📊 Using INDEX price")

else:
    print("❌ Price data not found")
    exit()

# ===============================
# DETECT OI SOURCE
# ===============================
oi_file = None

futstk_file = os.path.join(futstk_path, f"{symbol}.csv")
futidx_file = os.path.join(futidx_path, f"{symbol}.csv")

if os.path.exists(futstk_file):
    oi_file = futstk_file
    print("📊 Using STOCK Futures OI")

elif os.path.exists(futidx_file):
    oi_file = futidx_file
    print("📊 Using INDEX Futures OI")

else:
    print("⚠️ No OI data available")

# ===============================
# LOAD PRICE DATA
# ===============================
price_df = pd.read_csv(price_file)
price_df.columns = price_df.columns.str.upper()

if "CLOSE" not in price_df.columns:
    print("❌ CLOSE column missing")
    exit()

price_series = price_df["CLOSE"].dropna()

# safety sort
price_series = price_series.reset_index(drop=True)

current_price = price_series.iloc[-1]
price_pct = (price_series < current_price).sum() / len(price_series) * 100

print(f"\n📊 Price Percentile: {price_pct:.2f}")

# ===============================
# LOAD OI DATA (IF EXISTS)
# ===============================
oi_pct = None

if oi_file:
    oi_df = pd.read_csv(oi_file)
    oi_df.columns = oi_df.columns.str.upper()

    # ===============================
    # DETECT OI COLUMN
    # ===============================
    oi_col = None
    for col in ["OI", "OPEN_INT", "OPENINTEREST"]:
        if col in oi_df.columns:
            oi_col = col
            break

    if oi_col:

        # ===============================
        # FIX EXPIRY DATE (CRITICAL FIX)
        # ===============================
        if "EXP_DATE" in oi_df.columns:
            oi_df["EXP_DATE"] = pd.to_datetime(
                oi_df["EXP_DATE"],
                format="%Y%m%d",     # 🔥 FIX HERE
                errors="coerce"
            )

            oi_df = oi_df.dropna(subset=["EXP_DATE"])

            latest_expiry = oi_df["EXP_DATE"].max()
            oi_df = oi_df[oi_df["EXP_DATE"] == latest_expiry]

            print(f"📅 Using Expiry: {latest_expiry.date()}")

        # ===============================
        # CALCULATE OI PERCENTILE
        # ===============================
        oi_series = oi_df[oi_col].dropna().reset_index(drop=True)

        current_oi = oi_series.iloc[-1]
        oi_pct = (oi_series < current_oi).sum() / len(oi_series) * 100

        print(f"📊 OI Percentile: {oi_pct:.2f}")

# ===============================
# PLOT
# ===============================
fig, ax = plt.subplots(figsize=(5,6))
bar_width = 0.28

# -------- DUAL PANEL --------
if oi_pct is not None:

    # background
    ax.bar(-0.4, 100, width=bar_width, color='#64748b', alpha=0.3)
    ax.bar(0.4, 100, width=bar_width, color='#64748b', alpha=0.3)

    # fill
    ax.bar(-0.4, price_pct, width=bar_width, color='#34d399', edgecolor='#10b981', linewidth=1.5)
    ax.bar(0.4, oi_pct, width=bar_width, color='#38bdf8', edgecolor='#0ea5e9', linewidth=1.5)

    # glow
    ax.bar(-0.4, price_pct, width=bar_width, color='white', alpha=0.05)
    ax.bar(0.4, oi_pct, width=bar_width, color='white', alpha=0.05)

    # labels
    ax.text(-0.4, 105, "Price Percentile", ha='center', fontsize=10)
    ax.text(0.4, 105, "OI Percentile", ha='center', fontsize=10)

    ax.text(-0.4, -6, f"{price_pct:.1f}", ha='center', fontsize=13, fontweight='bold')
    ax.text(0.4, -6, f"{oi_pct:.1f}", ha='center', fontsize=13, fontweight='bold')

# -------- SINGLE PANEL --------
else:
    ax.bar(0, 100, width=bar_width, color='#64748b', alpha=0.3)
    ax.bar(0, price_pct, width=bar_width, color='#34d399', edgecolor='#10b981')

    ax.text(0, 105, "Price Percentile", ha='center', fontsize=10)
    ax.text(0, -6, f"{price_pct:.1f}", ha='center', fontsize=13, fontweight='bold')

# -------- COMMON UI --------
for level in [20, 40, 60, 80]:
    ax.hlines(level, -0.8, 0.8, colors='white', linewidth=1, alpha=0.2)

ax.text(0, 112, symbol, ha='center', fontsize=15, fontweight='bold')

ax.set_xlim(-1, 1)
ax.set_ylim(0, 110)
ax.axis('off')

plt.tight_layout()
plt.show()