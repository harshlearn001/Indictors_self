import pandas as pd 
import os 
from glob import glob 

# ================================ # PATH CONFIG # ================================ 
OPTION_PATH = r"H:\MarketForge\data\master\option_master" 
STOCK_SPOT_PATH = r"H:\MarketForge\data\master\Equity_stock_master" 
INDEX_SPOT_PATH = r"H:\MarketForge\data\master\Indices_master" 

# Separate Output Files
BULLISH_OUTPUT = r"H:\Indictors_self\output\final_signals_bullish.csv"
BEARISH_OUTPUT = r"H:\Indictors_self\output\final_signals_bearish.csv"

STRIKE_FILTER = 2.0  # 0–2% Max proximity from Spot
MIN_OI_CHANGE = 5.0  # Minimum % OI increase to filter noise

# ================================ # GET SPOT PRICE # ================================ 
def get_spot(symbol):
    try:
        f1 = os.path.join(STOCK_SPOT_PATH, f"{symbol}.csv")
        f2 = os.path.join(INDEX_SPOT_PATH, f"{symbol}.csv")
        if os.path.exists(f1):
            df = pd.read_csv(f1)
        elif os.path.exists(f2):
            df = pd.read_csv(f2)
        else:
            return None
        
        df.columns = df.columns.str.upper()
        return float(df.iloc[-1]["CLOSE"])
    except:
        return None

# ================================ # PROCESS EACH STOCK # ================================ 
def process_file(path):
    try:
        df = pd.read_parquet(path)
        df.columns = df.columns.str.upper()
        
        # 1. Sort to properly calculate historical diffs
        df = df.sort_values(["SYMBOL", "EXP_DATE", "STRIKE_PRICE", "OPT_TYPE", "TRADE_DATE"])
        
        # 2. Calculate OI % Change from previous day
        df["OI_CHANGE"] = df.groupby(["SYMBOL", "EXP_DATE", "STRIKE_PRICE", "OPT_TYPE"])["OPEN_INT"].diff()
        df["OI_CHANGE_PCT"] = (df["OI_CHANGE"] / df.groupby(["SYMBOL", "EXP_DATE", "STRIKE_PRICE", "OPT_TYPE"])["OPEN_INT"].shift(1)) * 100
        df = df.dropna(subset=["OI_CHANGE_PCT"])
        
        # 3. Filter for the latest Snapshot day
        latest = df["TRADE_DATE"].max()
        df = df[df["TRADE_DATE"] == latest]
        if df.empty:
            return None
            
        symbol = df["SYMBOL"].iloc[0]
        
        # Split into Calls and Puts
        calls = df[df["OPT_TYPE"] == "CE"]
        puts = df[df["OPT_TYPE"] == "PE"]
        if calls.empty or puts.empty:
            return None

        # 4. STEP 1: Find Max OI Strikes
        call_max = calls.loc[calls["OPEN_INT"].idxmax()]
        put_max = puts.loc[puts["OPEN_INT"].idxmax()]
        
        call_strike = call_max["STRIKE_PRICE"]
        put_strike = put_max["STRIKE_PRICE"]
        
        call_chg = call_max["OI_CHANGE_PCT"]
        put_chg = put_max["OI_CHANGE_PCT"]

        # 5. STEP 2: Fetch Spot Price
        spot = get_spot(symbol)
        if spot is None:
            return None

        # 6. STEP 3 & 4: Select within 2% window & verify growing OI
        put_diff = abs(spot - put_strike) / spot * 100
        call_diff = abs(call_strike - spot) / spot * 100
        
        signal = None
        
        # 🟢 BULLISH: Spot is near PUT Max OI Support, and PUT OI is increasing
        if (put_diff <= STRIKE_FILTER) and (put_chg >= MIN_OI_CHANGE):
            signal = "BULLISH"
            
        # 🔴 BEARISH: Spot is near CALL Max OI Resistance, and CALL OI is increasing
        elif (call_diff <= STRIKE_FILTER) and (call_chg >= MIN_OI_CHANGE):
            signal = "BEARISH"
            
        if signal is None:
            return None

        # Strength based on the active structural side's OI building speed
        strength = put_chg if signal == "BULLISH" else call_chg

        return {
            "SYMBOL": symbol,
            "SPOT": round(spot, 2),
            "PUT_STRIKE": put_strike,
            "PUT_DIFF_%": round(put_diff, 2),
            "PUT_OI_CHG_%": round(put_chg, 2),
            "CALL_STRIKE": call_strike,
            "CALL_DIFF_%": round(call_diff, 2),
            "CALL_OI_CHG_%": round(call_chg, 2),
            "SIGNAL": signal,
            "STRENGTH": round(strength, 2)
        }
    except Exception as e:
        print(f"Error in {os.path.basename(path)}: {e}")
        return None

# ================================ # MAIN ENGINE # ================================ 
def run():
    files = glob(os.path.join(OPTION_PATH, "**", "*.parquet"), recursive=True)
    print(f"\n📂 Total files found: {len(files)}")
    
    results = []
    for f in files:
        res = process_file(f)
        if res:
            results.append(res)
            
    if not results:
        print("\n⚠️ No structural signals match your 2% boundaries with increasing OI today.")
        return
        
    master_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(BULLISH_OUTPUT), exist_ok=True)

    # 🟢 Filter and Save Bullish
    bullish_df = master_df[master_df["SIGNAL"] == "BULLISH"].sort_values(by="STRENGTH", ascending=False)
    if not bullish_df.empty:
        bullish_df.to_csv(BULLISH_OUTPUT, index=False)
        print("\n🟢 BULLISH SIGNALS READY")
        print(f"📁 Saved at: {BULLISH_OUTPUT}")
        print(bullish_df.head(10))
    else:
        print("\n🟢 No Bullish signals found matching conditions.")

    # 🔴 Filter and Save Bearish
    bearish_df = master_df[master_df["SIGNAL"] == "BEARISH"].sort_values(by="STRENGTH", ascending=False)
    if not bearish_df.empty:
        bearish_df.to_csv(BEARISH_OUTPUT, index=False)
        print("\n🔴 BEARISH SIGNALS READY")
        print(f"📁 Saved at: {BEARISH_OUTPUT}")
        print(bearish_df.head(10))
    else:
        print("\n🔴 No Bearish signals found matching conditions.")

if __name__ == "__main__":
    run()
