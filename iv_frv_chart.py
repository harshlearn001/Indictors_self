import pandas as pd
from pathlib import Path

# ============================================================
# OPTIONGREEKS
# IV - FRV CHART
# PART 1
# ============================================================

print("=" * 70)
print("              OPTIONGREEKS IV - FRV CHART")
print("=" * 70)

SYMBOL = input("Enter Symbol            : ").strip().upper()

days = input("Number of Days [180]    : ").strip()

LAST_DAYS = int(days) if days else 180


# ============================================================
# SEARCH PATHS
# ============================================================

IV_PATHS = [

    Path(r"H:\OptionGreeks\data\greeks\STOCKS"),

    Path(r"H:\OptionGreeks\data\greeks\INDICES"),

]


# ============================================================
# FIND FILE
# ============================================================

def find_file(paths, symbol):

    for root in paths:

        files = list(root.rglob(f"{symbol}.csv"))

        if files:

            return files[0], root.name

    raise FileNotFoundError(f"{symbol}.csv not found")


iv_file, category = find_file(IV_PATHS, SYMBOL)

print("\nCategory :", category)

print("IV File  :", iv_file)


# ============================================================
# LOAD
# ============================================================

iv = pd.read_csv(iv_file)

print("\nRows :", len(iv))


# ============================================================
# DATE
# ============================================================

iv["TRADE_DATE"] = pd.to_datetime(

    iv["TRADE_DATE"].astype(str),

    format="%Y%m%d"

)

iv["EXP_DATE"] = pd.to_datetime(

    iv["EXP_DATE"].astype(str),

    format="%Y%m%d"

)


# ============================================================
# BUILD DAILY ATM IV
# ============================================================

rows = []

dates = sorted(iv["TRADE_DATE"].unique())

print("\nBuilding ATM IV ...")


for trade_date in dates:

    day = iv[

        iv["TRADE_DATE"] == trade_date

    ]

    if day.empty:

        continue

    # ------------------------------------
    # nearest expiry
    # ------------------------------------

    expiry = day["EXP_DATE"].min()

    day = day[

        day["EXP_DATE"] == expiry

    ]

    # ------------------------------------
    # spot
    # ------------------------------------

    spot = day["SPOT_CLOSE"].iloc[0]

    # ------------------------------------
    # nearest strike
    # ------------------------------------

    strikes = sorted(

        day["STRIKE_PRICE"].unique()

    )

    atm = min(

        strikes,

        key=lambda x: abs(x - spot)

    )

    atm = day[

        day["STRIKE_PRICE"] == atm

    ]

    ce = atm[

        atm["OPT_TYPE"] == "CE"

    ]

    pe = atm[

        atm["OPT_TYPE"] == "PE"

    ]

    if ce.empty or pe.empty:

        continue

    ce_iv = ce.iloc[0]["IMPLIED_VOL"]

    pe_iv = pe.iloc[0]["IMPLIED_VOL"]

    atm_iv = (

        ce_iv +

        pe_iv

    ) / 2

    rows.append(

        {

            "TRADE_DATE": trade_date,

            "ATM_IV": atm_iv,

            "SPOT": spot,

            "ATM_STRIKE": atm.iloc[0]["STRIKE_PRICE"]

        }

    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

if df.empty:

    raise Exception("ATM IV not calculated.")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(

    "TRADE_DATE"

).reset_index(

    drop=True

)


# ============================================================
# FRV
# ============================================================

#
# USER DEFINITION
#
# FRV = Today's IV - Yesterday's IV
#

df["FRV"] = (

    df["ATM_IV"]

    -

    df["ATM_IV"].shift(1)

) * 100


# ============================================================
# COLORS
# ============================================================

df["BAR_COLOR"] = df["FRV"].apply(

    lambda x:

    "#32ff32"

    if x >= 0

    else

    "#ff4040"

)


# ============================================================
# LAST N DAYS
# ============================================================

df = df.tail(

    LAST_DAYS

).reset_index(

    drop=True

)


print("\n")

print(df.tail())

print("\nCurrent")

print("----------------------------")

print("ATM IV :", round(df.iloc[-1]["ATM_IV"],4))

print("FRV    :", round(df.iloc[-1]["FRV"],2))

print("SPOT   :", round(df.iloc[-1]["SPOT"],2))

print("----------------------------")
# ============================================================
# PART 2
# SENSIBULL STYLE CHART
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("dark_background")

fig, ax1 = plt.subplots(figsize=(18,8))

# ------------------------------------------------------------
# Background
# ------------------------------------------------------------

fig.patch.set_facecolor("#061b33")
ax1.set_facecolor("#082340")

# ------------------------------------------------------------
# Zero Line
# ------------------------------------------------------------

ax1.axhline(
    0,
    color="white",
    linewidth=1,
    linestyle="--",
    alpha=0.70
)

# ------------------------------------------------------------
# Histogram
# ------------------------------------------------------------

positive = df["FRV"] >= 0
negative = df["FRV"] < 0

ax1.bar(
    df.loc[positive,"TRADE_DATE"],
    df.loc[positive,"FRV"],
    width=0.8,
    color="#4cff32",
    edgecolor="#4cff32"
)

ax1.bar(
    df.loc[negative,"TRADE_DATE"],
    df.loc[negative,"FRV"],
    width=0.8,
    color="#ff3a3a",
    edgecolor="#ff3a3a"
)

# ------------------------------------------------------------
# Left Axis
# ------------------------------------------------------------

ax1.set_ylabel(
    "IV - FRV",
    fontsize=12,
    color="white"
)

# ------------------------------------------------------------
# Right Axis
# ------------------------------------------------------------

ax2 = ax1.twinx()

ax2.plot(
    df["TRADE_DATE"],
    df["SPOT"],
    color="#f2a900",
    linewidth=1.8
)

ax2.set_ylabel(
    "Spot",
    color="white"
)

# ------------------------------------------------------------
# X Axis
# ------------------------------------------------------------

ax1.xaxis.set_major_locator(
    mdates.MonthLocator()
)

ax1.xaxis.set_major_formatter(
    mdates.DateFormatter("%b")
)

plt.setp(
    ax1.get_xticklabels(),
    rotation=0
)

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax1.grid(
    alpha=0.12,
    linestyle="--"
)

# ------------------------------------------------------------
# Current Values
# ------------------------------------------------------------

last = df.iloc[-1]

txt = (
    f"Date : {last.TRADE_DATE:%d-%b-%Y}\n\n"
    f"IV-FRV : {last.FRV:.2f}\n\n"
    f"Spot : {last.SPOT:.2f}"
)

ax1.text(

    0.89,

    0.88,

    txt,

    transform=ax1.transAxes,

    fontsize=11,

    color="white",

    bbox=dict(

        facecolor="#0b2c52",

        edgecolor="white",

        alpha=0.6

    )

)

# ------------------------------------------------------------
# Limits
# ------------------------------------------------------------

m = max(abs(df["FRV"].min()),abs(df["FRV"].max()))

ax1.set_ylim(

    -m*1.2,

     m*1.2

)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

plt.title(

    f"{SYMBOL} ({category})  IV - FRV",

    fontsize=18,

    color="white"

)

# ------------------------------------------------------------
# Cosmetics
# ------------------------------------------------------------

for s in ax1.spines.values():

    s.set_color("gray")

for s in ax2.spines.values():

    s.set_color("gray")

ax1.tick_params(colors="white")

ax2.tick_params(colors="white")

plt.tight_layout()

plt.show()