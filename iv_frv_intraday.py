import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# OPTIONGREEKS
# IV - FRV CHART
#
# FRV DEFINITION
#
# Morning IV = ATM IV calculated using OPEN_PRICE
# Evening IV = ATM IV calculated using CLOSE_PRICE
#
# FRV = Morning IV - Evening IV
#
# Positive FRV:
# Morning IV > Evening IV
#
# Negative FRV:
# Morning IV < Evening IV
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

    raise FileNotFoundError(
        f"{symbol}.csv not found"
    )


iv_file, category = find_file(
    IV_PATHS,
    SYMBOL
)

print("\nCategory :", category)
print("IV File  :", iv_file)


# ============================================================
# LOAD
# ============================================================

iv = pd.read_csv(
    iv_file,
    low_memory=False
)

print("\nRows :", len(iv))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "TRADE_DATE",
    "EXP_DATE",
    "STRIKE_PRICE",
    "OPT_TYPE",
    "OPEN_PRICE",
    "CLOSE_PRICE",
    "SPOT_CLOSE"

]

missing = [

    col

    for col in required_columns

    if col not in iv.columns

]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# DATE
# ============================================================

iv["TRADE_DATE"] = pd.to_datetime(

    iv["TRADE_DATE"].astype(str),

    format="%Y%m%d",

    errors="coerce"

)

iv["EXP_DATE"] = pd.to_datetime(

    iv["EXP_DATE"].astype(str),

    format="%Y%m%d",

    errors="coerce"

)


# ============================================================
# NUMERIC
# ============================================================

numeric_columns = [

    "STRIKE_PRICE",
    "OPEN_PRICE",
    "CLOSE_PRICE",
    "SPOT_CLOSE"

]

for col in numeric_columns:

    iv[col] = pd.to_numeric(

        iv[col],

        errors="coerce"

    )


# ============================================================
# REMOVE BAD ROWS
# ============================================================

iv = iv.dropna(

    subset=[

        "TRADE_DATE",
        "EXP_DATE",
        "STRIKE_PRICE",
        "OPT_TYPE",
        "SPOT_CLOSE"

    ]

).copy()


# ============================================================
# ATM IV FUNCTION
#
# Calculates ATM IV from either:
#
# OPEN_PRICE
#
# or
#
# CLOSE_PRICE
#
# ============================================================

def calculate_atm_iv(day, price_column):

    if day.empty:

        return None


    # --------------------------------------------------------
    # nearest expiry
    # --------------------------------------------------------

    expiry = day["EXP_DATE"].min()

    day = day[

        day["EXP_DATE"] == expiry

    ].copy()

    if day.empty:

        return None


    # --------------------------------------------------------
    # spot
    # --------------------------------------------------------

    spot_values = (

        day["SPOT_CLOSE"]

        .dropna()

    )

    if spot_values.empty:

        return None


    spot = spot_values.iloc[0]

    if pd.isna(spot) or spot <= 0:

        return None


    # --------------------------------------------------------
    # nearest ATM strike
    # --------------------------------------------------------

    strikes = (

        day["STRIKE_PRICE"]

        .dropna()

        .unique()

    )

    if len(strikes) == 0:

        return None


    atm_strike = min(

        strikes,

        key=lambda x: abs(x - spot)

    )


    # --------------------------------------------------------
    # ATM STRIKE
    # --------------------------------------------------------

    atm = day[

        day["STRIKE_PRICE"] == atm_strike

    ].copy()


    # --------------------------------------------------------
    # CE
    # --------------------------------------------------------

    ce = atm[

        atm["OPT_TYPE"]

        .astype(str)

        .str.upper()

        == "CE"

    ].copy()


    # --------------------------------------------------------
    # PE
    # --------------------------------------------------------

    pe = atm[

        atm["OPT_TYPE"]

        .astype(str)

        .str.upper()

        == "PE"

    ].copy()


    if ce.empty or pe.empty:

        return None


    # --------------------------------------------------------
    # OPTION PRICES
    # --------------------------------------------------------

    ce_price = pd.to_numeric(

        ce[price_column],

        errors="coerce"

    ).dropna()


    pe_price = pd.to_numeric(

        pe[price_column],

        errors="coerce"

    ).dropna()


    if ce_price.empty or pe_price.empty:

        return None


    # --------------------------------------------------------
    # IMPORTANT
    #
    # The existing IMPLIED_VOL column in OptionGreeks was
    # calculated from CLOSE_PRICE.
    #
    # Therefore:
    #
    # Morning IV:
    # We need an IV corresponding to OPEN_PRICE.
    #
    # Evening IV:
    # Use existing IMPLIED_VOL calculated from CLOSE_PRICE.
    #
    # --------------------------------------------------------

    if price_column == "CLOSE_PRICE":

        ce_iv = pd.to_numeric(

            ce["IMPLIED_VOL"],

            errors="coerce"

        ).dropna()


        pe_iv = pd.to_numeric(

            pe["IMPLIED_VOL"],

            errors="coerce"

        ).dropna()


        if ce_iv.empty or pe_iv.empty:

            return None


        ce_iv_value = ce_iv.median()

        pe_iv_value = pe_iv.median()


    else:

        # ----------------------------------------------------
        # MORNING IV
        #
        # We cannot simply use the daily IMPLIED_VOL here
        # because that IV was calculated from CLOSE_PRICE.
        #
        # We therefore calculate IV from OPEN_PRICE using
        # Black-Scholes.
        #
        # ----------------------------------------------------

        raise RuntimeError(

            "OPEN_PRICE IV requires Black-Scholes "
            "calculation. Use the BS section below."

        )


    atm_iv = (

        ce_iv_value +

        pe_iv_value

    ) / 2.0


    return {

        "ATM_IV": atm_iv,

        "SPOT": spot,

        "ATM_STRIKE": atm_strike,

        "EXPIRY": expiry

    }


# ============================================================
# BLACK-SCHOLES IV
# ============================================================

from scipy.optimize import brentq
from scipy.stats import norm


# ============================================================
# BLACK-SCHOLES PRICE
# ============================================================

def black_scholes_price(

    spot,
    strike,
    time,
    rate,
    dividend,
    volatility,
    option_type

):

    if (

        spot <= 0

        or strike <= 0

        or time <= 0

        or volatility <= 0

    ):

        return np.nan


    d1 = (

        np.log(spot / strike)

        + (

            rate

            - dividend

            + 0.5 * volatility ** 2

        ) * time

    ) / (

        volatility * np.sqrt(time)

    )


    d2 = (

        d1

        - volatility * np.sqrt(time)

    )


    if option_type == "CE":

        price = (

            spot

            * np.exp(-dividend * time)

            * norm.cdf(d1)

            -

            strike

            * np.exp(-rate * time)

            * norm.cdf(d2)

        )

    else:

        price = (

            strike

            * np.exp(-rate * time)

            * norm.cdf(-d2)

            -

            spot

            * np.exp(-dividend * time)

            * norm.cdf(-d1)

        )


    return price


# ============================================================
# IMPLIED VOLATILITY
# ============================================================

def calculate_iv(

    market_price,
    spot,
    strike,
    time,
    rate,
    dividend,
    option_type

):

    if (

        pd.isna(market_price)

        or pd.isna(spot)

        or pd.isna(strike)

        or pd.isna(time)

        or time <= 0

        or market_price <= 0

        or spot <= 0

        or strike <= 0

    ):

        return np.nan


    option_type = str(

        option_type

    ).upper()


    # --------------------------------------------------------
    # no-arbitrage bounds
    # --------------------------------------------------------

    if option_type == "CE":

        intrinsic = max(

            spot * np.exp(-dividend * time)

            -

            strike * np.exp(-rate * time),

            0

        )

    else:

        intrinsic = max(

            strike * np.exp(-rate * time)

            -

            spot * np.exp(-dividend * time),

            0

        )


    if market_price < intrinsic:

        return np.nan


    # --------------------------------------------------------
    # objective
    # --------------------------------------------------------

    def objective(vol):

        price = black_scholes_price(

            spot,

            strike,

            time,

            rate,

            dividend,

            vol,

            option_type

        )

        return price - market_price


    try:

        return brentq(

            objective,

            1e-6,

            5.0,

            maxiter=100

        )

    except Exception:

        return np.nan


# ============================================================
# CALCULATE MORNING ATM IV
# ============================================================

def calculate_morning_atm_iv(day):

    if day.empty:

        return None


    # --------------------------------------------------------
    # nearest expiry
    # --------------------------------------------------------

    expiry = day["EXP_DATE"].min()

    day = day[

        day["EXP_DATE"] == expiry

    ].copy()


    if day.empty:

        return None


    # --------------------------------------------------------
    # spot
    # --------------------------------------------------------

    spot_values = day[

        "SPOT_CLOSE"

    ].dropna()


    if spot_values.empty:

        return None


    spot = spot_values.iloc[0]


    # --------------------------------------------------------
    # ATM strike
    # --------------------------------------------------------

    strikes = (

        day["STRIKE_PRICE"]

        .dropna()

        .unique()

    )


    if len(strikes) == 0:

        return None


    atm_strike = min(

        strikes,

        key=lambda x: abs(x - spot)

    )


    atm = day[

        day["STRIKE_PRICE"] == atm_strike

    ].copy()


    ce = atm[

        atm["OPT_TYPE"]

        .astype(str)

        .str.upper()

        == "CE"

    ]


    pe = atm[

        atm["OPT_TYPE"]

        .astype(str)

        .str.upper()

        == "PE"

    ]


    if ce.empty or pe.empty:

        return None


    # --------------------------------------------------------
    # TIME TO EXPIRY
    # --------------------------------------------------------

    days_to_expiry = (

        expiry -

        day["TRADE_DATE"].iloc[0]

    ).days


    time = days_to_expiry / 365.0


    if time <= 0:

        return None


    # --------------------------------------------------------
    # RATE
    # --------------------------------------------------------

    if "RISK_FREE_RATE" in day.columns:

        rate_values = pd.to_numeric(

            day["RISK_FREE_RATE"],

            errors="coerce"

        ).dropna()

        rate = (

            rate_values.iloc[0]

            if not rate_values.empty

            else 0.06

        )

    else:

        rate = 0.06


    # --------------------------------------------------------
    # DIVIDEND
    # --------------------------------------------------------

    if "DIVIDEND_YIELD" in day.columns:

        div_values = pd.to_numeric(

            day["DIVIDEND_YIELD"],

            errors="coerce"

        ).dropna()

        dividend = (

            div_values.iloc[0]

            if not div_values.empty

            else 0.0

        )

    else:

        dividend = 0.0


    # --------------------------------------------------------
    # CE IV
    # --------------------------------------------------------

    ce_price = pd.to_numeric(

        ce["OPEN_PRICE"],

        errors="coerce"

    ).dropna()


    pe_price = pd.to_numeric(

        pe["OPEN_PRICE"],

        errors="coerce"

    ).dropna()


    if ce_price.empty or pe_price.empty:

        return None


    ce_open = ce_price.median()

    pe_open = pe_price.median()


    ce_iv = calculate_iv(

        market_price=ce_open,

        spot=spot,

        strike=atm_strike,

        time=time,

        rate=rate,

        dividend=dividend,

        option_type="CE"

    )


    pe_iv = calculate_iv(

        market_price=pe_open,

        spot=spot,

        strike=atm_strike,

        time=time,

        rate=rate,

        dividend=dividend,

        option_type="PE"

    )


    if pd.isna(ce_iv) or pd.isna(pe_iv):

        return None


    atm_iv = (

        ce_iv +

        pe_iv

    ) / 2.0


    return {

        "ATM_IV": atm_iv,

        "SPOT": spot,

        "ATM_STRIKE": atm_strike,

        "EXPIRY": expiry,

        "CE_IV": ce_iv,

        "PE_IV": pe_iv

    }


# ============================================================
# CALCULATE EVENING ATM IV
# ============================================================

def calculate_evening_atm_iv(day):

    if day.empty:

        return None


    expiry = day["EXP_DATE"].min()


    day = day[

        day["EXP_DATE"] == expiry

    ].copy()


    if day.empty:

        return None


    spot_values = day[

        "SPOT_CLOSE"

    ].dropna()


    if spot_values.empty:

        return None


    spot = spot_values.iloc[0]


    strikes = (

        day["STRIKE_PRICE"]

        .dropna()

        .unique()

    )


    if len(strikes) == 0:

        return None


    atm_strike = min(

        strikes,

        key=lambda x: abs(x - spot)

    )


    atm = day[

        day["STRIKE_PRICE"] == atm_strike

    ].copy()


    ce = atm[

        atm["OPT_TYPE"]

        .astype(str)

        .str.upper()

        == "CE"

    ]


    pe = atm[

        atm["OPT_TYPE"]

        .astype(str)

        .str.upper()

        == "PE"

    ]


    if ce.empty or pe.empty:

        return None


    ce_iv = pd.to_numeric(

        ce["IMPLIED_VOL"],

        errors="coerce"

    ).dropna()


    pe_iv = pd.to_numeric(

        pe["IMPLIED_VOL"],

        errors="coerce"

    ).dropna()


    if ce_iv.empty or pe_iv.empty:

        return None


    ce_iv_value = ce_iv.median()

    pe_iv_value = pe_iv.median()


    atm_iv = (

        ce_iv_value +

        pe_iv_value

    ) / 2.0


    return {

        "ATM_IV": atm_iv,

        "SPOT": spot,

        "ATM_STRIKE": atm_strike,

        "EXPIRY": expiry,

        "CE_IV": ce_iv_value,

        "PE_IV": pe_iv_value

    }


# ============================================================
# BUILD DAILY FRV
# ============================================================

rows = []

dates = sorted(

    iv["TRADE_DATE"]

    .dropna()

    .unique()

)


print("\nBuilding Morning / Evening ATM IV ...")


for trade_date in dates:

    day = iv[

        iv["TRADE_DATE"] == trade_date

    ].copy()


    if day.empty:

        continue


    # --------------------------------------------------------
    # MORNING
    # --------------------------------------------------------

    morning = calculate_morning_atm_iv(

        day

    )


    if morning is None:

        continue


    # --------------------------------------------------------
    # EVENING
    # --------------------------------------------------------

    evening = calculate_evening_atm_iv(

        day

    )


    if evening is None:

        continue


    # --------------------------------------------------------
    # FRV
    # --------------------------------------------------------

    frv = (

        morning["ATM_IV"]

        -

        evening["ATM_IV"]

    ) * 100.0


    rows.append({

        "TRADE_DATE": trade_date,

        "MORNING_IV": morning["ATM_IV"] * 100.0,

        "EVENING_IV": evening["ATM_IV"] * 100.0,

        "FRV": frv,

        "MORNING_CE_IV": morning["CE_IV"] * 100.0,

        "MORNING_PE_IV": morning["PE_IV"] * 100.0,

        "EVENING_CE_IV": evening["CE_IV"] * 100.0,

        "EVENING_PE_IV": evening["PE_IV"] * 100.0,

        "SPOT": evening["SPOT"],

        "ATM_STRIKE": evening["ATM_STRIKE"],

        "EXPIRY": evening["EXPIRY"]

    })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


if df.empty:

    raise Exception(

        "Morning / Evening ATM IV could not be calculated."

    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(

    "TRADE_DATE"

).reset_index(

    drop=True

)


# ============================================================
# LAST N DAYS
# ============================================================

df = df.tail(

    LAST_DAYS

).reset_index(

    drop=True

)


# ============================================================
# BAR COLOR
# ============================================================

df["BAR_COLOR"] = np.where(

    df["FRV"] >= 0,

    "#32ff32",

    "#ff4040"

)


# ============================================================
# PRINT DATA
# ============================================================

print("\n")
print("=" * 70)
print("                    FRV DATA")
print("=" * 70)


display_columns = [

    "TRADE_DATE",

    "MORNING_IV",

    "EVENING_IV",

    "FRV",

    "SPOT",

    "ATM_STRIKE"

]


print(

    df[display_columns]

    .tail(10)

    .to_string(index=False)

)


# ============================================================
# CURRENT
# ============================================================

last = df.iloc[-1]


print("\n")
print("CURRENT")
print("----------------------------")

print(

    "Date       :",

    last["TRADE_DATE"].strftime(

        "%d-%b-%Y"

    )

)

print(

    "Morning IV :",

    round(

        last["MORNING_IV"],

        4

    )

)

print(

    "Evening IV :",

    round(

        last["EVENING_IV"],

        4

    )

)

print(

    "FRV        :",

    round(

        last["FRV"],

        2

    )

)

print(

    "Spot       :",

    round(

        last["SPOT"],

        2

    )

)

print(

    "ATM Strike :",

    last["ATM_STRIKE"]

)

print("----------------------------")


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("FRV INTERPRETATION")
print("----------------------------")


if last["FRV"] > 0:

    print(

        "POSITIVE : Morning IV > Evening IV"

    )

elif last["FRV"] < 0:

    print(

        "NEGATIVE : Morning IV < Evening IV"

    )

else:

    print(

        "NEUTRAL  : Morning IV = Evening IV"

    )


# ============================================================
# PART 2
# SENSIBULL STYLE CHART
# ============================================================

plt.style.use(

    "dark_background"

)


fig, ax1 = plt.subplots(

    figsize=(18, 8)

)


# ============================================================
# BACKGROUND
# ============================================================

fig.patch.set_facecolor(

    "#061b33"

)

ax1.set_facecolor(

    "#082340"

)


# ============================================================
# ZERO LINE
# ============================================================

ax1.axhline(

    0,

    color="white",

    linewidth=1,

    linestyle="--",

    alpha=0.70

)


# ============================================================
# HISTOGRAM
# ============================================================

positive = df["FRV"] >= 0

negative = df["FRV"] < 0


ax1.bar(

    df.loc[positive, "TRADE_DATE"],

    df.loc[positive, "FRV"],

    width=0.8,

    color="#4cff32",

    edgecolor="#4cff32"

)


ax1.bar(

    df.loc[negative, "TRADE_DATE"],

    df.loc[negative, "FRV"],

    width=0.8,

    color="#ff3a3a",

    edgecolor="#ff3a3a"

)


# ============================================================
# LEFT AXIS
# ============================================================

ax1.set_ylabel(

    "FRV  (Morning IV - Evening IV)",

    fontsize=12,

    color="white"

)


# ============================================================
# RIGHT AXIS
# ============================================================

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


# ============================================================
# X AXIS
# ============================================================

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


# ============================================================
# GRID
# ============================================================

ax1.grid(

    alpha=0.12,

    linestyle="--"

)


# ============================================================
# CURRENT VALUES
# ============================================================

txt = (

    f"Date : {last.TRADE_DATE:%d-%b-%Y}\n\n"

    f"Morning IV : {last.MORNING_IV:.2f}%\n\n"

    f"Evening IV : {last.EVENING_IV:.2f}%\n\n"

    f"FRV : {last.FRV:.2f}\n\n"

    f"Spot : {last.SPOT:.2f}"

)


ax1.text(

    0.84,

    0.78,

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


# ============================================================
# LIMITS
# ============================================================

m = max(

    abs(df["FRV"].min()),

    abs(df["FRV"].max())

)


if m == 0:

    m = 1


ax1.set_ylim(

    -m * 1.2,

    m * 1.2

)


# ============================================================
# TITLE
# ============================================================

plt.title(

    f"{SYMBOL} ({category})  Morning IV - Evening IV (FRV)",

    fontsize=18,

    color="white"

)


# ============================================================
# COSMETICS
# ============================================================

for spine in ax1.spines.values():

    spine.set_color("gray")


for spine in ax2.spines.values():

    spine.set_color("gray")


ax1.tick_params(

    colors="white"

)

ax2.tick_params(

    colors="white"

)


plt.tight_layout()


# ============================================================
# SHOW
# ============================================================

plt.show()