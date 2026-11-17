import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# PATHS
# =====================================================
BASE_PATH = Path(r"H:\MarketForge\data\master\Futures_master_three_expiries")

FUTSTK_PATH = BASE_PATH / "FUTSTK"
FUTIDX_PATH = BASE_PATH / "FUTIDX"


# =====================================================
# LOAD DATA
# =====================================================
def load_data(symbol):

    file = None

    if (FUTSTK_PATH / f"{symbol}.csv").exists():
        file = FUTSTK_PATH / f"{symbol}.csv"
        print(f"📊 STOCK: {symbol}")

    elif (FUTIDX_PATH / f"{symbol}.csv").exists():
        file = FUTIDX_PATH / f"{symbol}.csv"
        print(f"📊 INDEX: {symbol}")

    else:
        print("❌ Symbol not found")
        return pd.DataFrame()

    df = pd.read_csv(file)
    df.columns = df.columns.str.upper()

    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'], format='%Y%m%d')
    df['EXP_DATE'] = pd.to_datetime(df['EXP_DATE'], format='%Y%m%d')

    return df


# =====================================================
# GENERIC OI PREP (FOR ANY EXPIRY)
# =====================================================
def prepare_expiry_oi(df, expiry_type):

    df = df.sort_values('TRADE_DATE')

    # last 25 trading days
    last_dates = df['TRADE_DATE'].drop_duplicates().sort_values().tail(25)
    df = df[df['TRADE_DATE'].isin(last_dates)]

    records = []

    for date, group in df.groupby('TRADE_DATE'):

        g = group[group['EXPIRY_TYPE'] == expiry_type]

        if g.empty:
            continue

        # dominant contract (max OI)
        row = g.loc[g['OPEN_INT'].idxmax()]

        records.append({
            "TRADE_DATE": date,
            "OPEN_INT": row['OPEN_INT'],
            "CLOSE_PRICE": row['CLOSE_PRICE']
        })

    return pd.DataFrame(records).sort_values('TRADE_DATE')


# =====================================================
# CLASSIFICATION
# =====================================================
def classify(df):

    df['PRICE_CHANGE'] = df['CLOSE_PRICE'].diff()
    df['OI_CHANGE'] = df['OPEN_INT'].diff()

    colors = []

    for i in range(len(df)):

        pc = df['PRICE_CHANGE'].iloc[i]
        oi = df['OI_CHANGE'].iloc[i]

        if pd.isna(pc) or pd.isna(oi):
            colors.append('gray')

        elif pc > 0 and oi > 0:
            colors.append('#006d5b')  # Long buildup

        elif pc < 0 and oi > 0:
            colors.append('#ff4d6d')  # Short buildup

        elif pc > 0 and oi < 0:
            colors.append('#66c2a5')  # Short covering

        else:
            colors.append('#f6b44b')  # Long unwinding

    df['COLOR'] = colors

    return df


# =====================================================
# PLOT FUNCTION
# =====================================================
def plot_expiry(df, symbol, expiry_type):

    df = prepare_expiry_oi(df, expiry_type)

    if df.empty:
        print(f"❌ No data for {expiry_type}")
        return

    df = classify(df)

    x = list(range(len(df)))

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # OI Bars
    ax1.bar(x, df['OPEN_INT'] / 1e6, color=df['COLOR'], width=0.85)
    ax1.set_ylabel("OI (Millions)")

    # Price Line
    ax2 = ax1.twinx()
    ax2.plot(x, df['CLOSE_PRICE'], linewidth=2)
    ax2.set_ylabel("Price")

    step = max(len(df)//10, 1)

    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(
        df['TRADE_DATE'].dt.strftime('%d-%b').iloc[::step],
        rotation=45
    )

    plt.title(f"{symbol} - {expiry_type} EXPIRY OI")

    plt.tight_layout()
    plt.show()


# =====================================================
# MULTI EXPIRY PLOT
# =====================================================
def plot_all_expiries(df, symbol):

    print("\n🚀 Plotting NEAR EXPIRY")
    plot_expiry(df, symbol, "NEAR")

    print("\n🚀 Plotting NEXT EXPIRY")
    plot_expiry(df, symbol, "NEXT")

    print("\n🚀 Plotting FAR EXPIRY")
    plot_expiry(df, symbol, "FAR")


# =====================================================
# MAIN
# =====================================================
def main():

    print("\n==============================")
    print("   FUTURES OI ANALYZER PRO")
    print("==============================")

    while True:

        symbol = input("\n👉 Enter Stock/Index (or EXIT): ").strip().upper()

        if symbol == "EXIT":
            break

        df = load_data(symbol)

        if not df.empty:
            plot_all_expiries(df, symbol)


# =====================================================
if __name__ == "__main__":
    main()


