import yfinance as yf

import pandas as pd

import numpy as np

from datetime import datetime

TICKERS = [

    "AAPL",

    "MSFT",

    "NVDA",

    "META",

    "IBM",

    "AMZN",

    "GOOGL",

    "AMD",

    "MU",

]

BENCHMARK = "^GSPC"

def download_data(ticker, period="20y"):

    data = yf.download(

        ticker,

        period=period,

        auto_adjust=True,

        progress=False

    )

    if data.empty:

        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    return data.dropna()

def september_decline_probability(ticker):

    data = download_data(ticker, period="max")

    if data.empty:

        return pd.DataFrame()

    data["Return"] = data["Close"].pct_change()

    september = data[data.index.month == 9].copy()

    september["Day"] = september.index.day

    september["Down"] = september["Return"] < 0

    result = (

        september.groupby("Day")

        .agg(

            observations=("Down", "count"),

            down_days=("Down", "sum"),

            average_return=("Return", "mean"),

        )

    )

    result["decline_probability"] = (

        result["down_days"] / result["observations"] * 100

    )

    result["average_return"] *= 100

    return result.round(2)

def moving_average_cycle(data, window):

    if len(data) < window:

        return np.nan

    ma = data["Close"].rolling(window).mean()

    current_price = float(data["Close"].iloc[-1])

    current_ma = float(ma.iloc[-1])

    return ((current_price / current_ma) - 1) * 100

def calculate_cycles(ticker):

    data = download_data(ticker)

    if data.empty:

        return {}

    return {

        "40_day_cycle": moving_average_cycle(data, 40),

        "100_day_cycle": moving_average_cycle(data, 100),

        "300_day_cycle": moving_average_cycle(data, 300),

    }

def cycle_risk_score(cycles):

    risk = 0

    for value in cycles.values():

        if pd.isna(value):

            continue

        if value > 10:

            risk += 20

        elif value > 5:

            risk += 12

        elif value > 0:

            risk += 5

        elif value < -10:

            risk += 15

    return min(risk, 60)

def september_risk_score(ticker):

    table = september_decline_probability(ticker)

    if table.empty:

        return 0

    high_risk_days = table[

        table["decline_probability"] >= 55

    ]

    if high_risk_days.empty:

        return 0

    avg_probability = high_risk_days[

        "decline_probability"

    ].mean()

    return min(avg_probability * 0.4, 40)

def market_risk_score(ticker):

    cycles = calculate_cycles(ticker)

    score = (

        cycle_risk_score(cycles)

        + september_risk_score(ticker)

    )

    return round(min(score, 100), 1)

def risk_label(score):

    if score >= 75:

        return "VERY HIGH"

    elif score >= 60:

        return "HIGH"

    elif score >= 40:

        return "ELEVATED"

    elif score >= 20:

        return "MODERATE"

    else:

        return "LOW"

def analyze_stock(ticker):

    cycles = calculate_cycles(ticker)

    score = market_risk_score(ticker)

    return {

        "Ticker": ticker,

        "40-Day Cycle %": round(cycles.get("40_day_cycle", np.nan), 2),

        "100-Day Cycle %": round(cycles.get("100_day_cycle", np.nan), 2),

        "300-Day Cycle %": round(cycles.get("300_day_cycle", np.nan), 2),

        "Risk Score": score,

        "Risk Level": risk_label(score),

    }

def analyze_market():

    results = []

    for ticker in TICKERS:

        try:

            results.append(analyze_stock(ticker))

        except Exception as error:

            print(f"Error analyzing {ticker}: {error}")

    return pd.DataFrame(results)

if __name__ == "__main__":

    print("\nSEPTEMBER MARKET RISK ANALYZER")

    print("=" * 50)

    market = analyze_market()

    print(market.to_string(index=False))

    print("\nSeptember S&P 500 decline probabilities:")

    print(

        september_decline_probability(BENCHMARK)

        .sort_values(

            "decline_probability",

            ascending=False

        )

        .head(10)

    )

    print(

        "\nAnalysis generated:",

        datetime.now().strftime("%Y-%m-%d %H:%M")

    )
