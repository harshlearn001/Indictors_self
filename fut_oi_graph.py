import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta

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
# CURRENT + NEXT OI (SENSIBULL PERFECT LOGIC)
# =====================================================
def prepare_current_next_oi(df):

    # Sort properly
    df = df.sort_values('TRADE_DATE')

    # 🔥 FIX: take last 25 TRADING DAYS (not calendar)
    last_dates = df['TRADE_DATE'].drop_duplicates().sort_values().tail(25)

    df = df[df['TRADE_DATE'].isin(last_dates)]

    records = []

    for date, group in df.groupby('TRADE_DATE'):

        near_group = group[group['EXPIRY_TYPE'] == 'NEAR']
        next_group = group[group['EXPIRY_TYPE'] == 'NEXT']

        if near_group.empty or next_group.empty:
            continue

        # select dominant contract
        near = near_group.loc[near_group['OPEN_INT'].idxmax()]
        next_ = next_group.loc[next_group['OPEN_INT'].idxmax()]

        total_oi = near['OPEN_INT'] + next_['OPEN_INT']
        price = near['CLOSE_PRICE']

        records.append({
            "TRADE_DATE": date,
            "OPEN_INT": total_oi,
            "CLOSE_PRICE": price
        })

    df_new = pd.DataFrame(records)

    return df_new.sort_values('TRADE_DATE')
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
# PLOT
# =====================================================
def plot_current_next(df, symbol):

    df = prepare_current_next_oi(df)

    if df.empty:
        print("❌ No data after processing")
        return

    df = classify(df)

    # Equal spacing (IMPORTANT FIX)
    x = list(range(len(df)))

    fig, ax1 = plt.subplots(figsize=(14,7))

    # OI Bars
    ax1.bar(x, df['OPEN_INT']/1e6, color=df['COLOR'], width=0.85)
    ax1.set_ylabel("OI (Millions)")

    # Price Line
    ax2 = ax1.twinx()
    ax2.plot(x, df['CLOSE_PRICE'], linewidth=2)
    ax2.set_ylabel("Price")

    # Clean X-axis
    step = max(len(df)//12, 1)

    ax1.set_xticks(x[::step])
    ax1.set_xticklabels(
        df['TRADE_DATE'].dt.strftime('%d-%b').iloc[::step],
        rotation=45
    )

    plt.title(f"{symbol} - Current + Next OI (Sensibull Style)")

    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN
# =====================================================
def main():

    print("\n==============================")
    print("   FUTURES OI ANALYZER")
    print("==============================")

    while True:

        symbol = input("\n👉 Enter Stock/Index (or EXIT): ").strip().upper()

        if symbol == "EXIT":
            break

        df = load_data(symbol)

        if not df.empty:
            plot_current_next(df, symbol)


# =====================================================
if __name__ == "__main__":
    main()