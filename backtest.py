import os
import math
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================
# SETTINGS
# =========================

START_CAPITAL = 50_000.0
RISK_PER_TRADE = 0.0075       # %0.75
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.20
MAX_INVESTED_PCT = 0.70

COMMISSION = 0.001            # %0.10
SLIPPAGE = 0.001              # %0.10
ATR_MULTIPLIER = 2.0

START_DATE = "2019-01-01"

SYMBOLS = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR",
    "BIMAS", "BRSAN", "CCOLA", "DOAS", "DOHOL",
    "ECILC", "EKGYO", "ENKAI", "EREGL", "FENER",
    "FROTO", "GARAN", "GUBRF", "HALKB", "HEKTS",
    "ISCTR", "KCHOL", "KONTR", "KOZAA", "KOZAL",
    "KRDMD", "MGROS", "OYAKC", "PETKM", "PGSUS",
    "SAHOL", "SASA", "SISE", "SKBNK", "TAVHL",
    "TCELL", "THYAO", "TKFEN", "TOASO", "TUPRS",
    "ULKER", "VAKBN", "YKBNK"
]


# =========================
# INDICATORS
# =========================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def prepare_data(df):

    df = df.copy()

    if df.empty:
        return df

    df = df.dropna()

    df["EMA20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50,
        adjust=False
    ).mean()

    df["RSI"] = calculate_rsi(df["Close"], 14)

    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # IMPORTANT:
    # Previous 20-day high, excluding today's candle.
    df["PREV_20_HIGH"] = df["High"].rolling(20).max().shift(1)

    df["RETURN_20"] = (
        df["Close"] / df["Close"].shift(20) - 1
    )

    # Signal is calculated using today's completed candle.
    conditions = pd.DataFrame(index=df.index)

    conditions["trend"] = df["EMA20"] > df["EMA50"]

    conditions["rsi"] = (
        (df["RSI"] >= 50) &
        (df["RSI"] <= 72)
    )

    conditions["volume"] = (
        df["Volume"] > df["VOL_MA20"] * 1.20
    )

    conditions["breakout"] = (
        df["Close"] > df["PREV_20_HIGH"]
    )

    conditions["momentum"] = (
        df["RETURN_20"] > 0.05
    )

    df["SIGNAL_SCORE"] = conditions.sum(axis=1)

    df["SIGNAL"] = df["SIGNAL_SCORE"] >= 4

    return df


# =========================
# DOWNLOAD DATA
# =========================

def download_data():

    data = {}

    print("Downloading BIST data...")

    for i, symbol in enumerate(SYMBOLS, 1):

        ticker = symbol + ".IS"

        print(
            f"[{i}/{len(SYMBOLS)}] {ticker}"
        )

        try:

            df = yf.download(
                ticker,
                start=START_DATE,
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                print("  No data.")
                continue

            # yfinance can sometimes return MultiIndex columns.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]

            if not all(
                col in df.columns
                for col in required
            ):
                print("  Missing columns.")
                continue

            df = df[required].copy()

            df = df.dropna()

            df = prepare_data(df)

            if len(df) < 100:
                print("  Not enough data.")
                continue

            data[symbol] = df

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

    print(
        f"\nLoaded {len(data)} symbols."
    )

    return data


# =========================
# BACKTEST
# =========================

def run_backtest(data):

    all_dates = set()

    for df in data.values():

        all_dates.update(
            df.index.tolist()
        )

    all_dates = sorted(all_dates)

    cash = START_CAPITAL

    positions = {}

    trades = []

    equity_records = []

    portfolio_value = START_CAPITAL

    for date in all_dates:

        # ---------------------------------
        # 1. EXIT POSITIONS
        # ---------------------------------

        symbols_to_close = []

        for symbol, pos in positions.items():

            df = data[symbol]

            if date not in df.index:
                continue

            idx = df.index.get_loc(date)

            if idx < 1:
                continue

            row = df.iloc[idx]

            previous = df.iloc[idx - 1]

            # Exit conditions use previous day's information.
            signal_break = (
                previous["EMA20"]
                <= previous["EMA50"]
            )

            momentum_loss = (
                previous["RSI"] < 45
            )

            stop_price = pos["stop_price"]

            today_open = row["Open"]
            today_low = row["Low"]

            exit_price = None
            exit_reason = None

            # Stop checked first.
            if today_open <= stop_price:

                exit_price = today_open
                exit_reason = "STOP_GAP"

            elif today_low <= stop_price:

                exit_price = stop_price
                exit_reason = "STOP"

            elif signal_break or momentum_loss:

                exit_price = today_open
                exit_reason = "SIGNAL_EXIT"

            if exit_price is not None:

                # Slippage on exit.
                exit_price *= (
                    1 - SLIPPAGE
                )

                gross_value = (
                    pos["shares"] * exit_price
                )

                commission = (
                    gross_value * COMMISSION
                )

                net_value = (
                    gross_value - commission
                )

                cash += net_value

                pnl = (
                    net_value
                    - pos["cost"]
                )

                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "side": "SELL",
                    "price": exit_price,
                    "shares": pos["shares"],
                    "value": net_value,
                    "pnl": pnl,
                    "reason": exit_reason
                })

                symbols_to_close.append(symbol)

        for symbol in symbols_to_close:

            del positions[symbol]


        # ---------------------------------
        # 2. CALCULATE CURRENT EQUITY
        # ---------------------------------

        invested_value = 0.0

        for symbol, pos in positions.items():

            df = data[symbol]

            if date in df.index:

                close_price = df.loc[
                    date,
                    "Close"
                ]

                if isinstance(close_price, pd.Series):
                    close_price = close_price.iloc[0]

                invested_value += (
                    pos["shares"] * float(close_price)
                )

        portfolio_value = (
            cash + invested_value
        )


        # ---------------------------------
        # 3. NEW ENTRIES
        # ---------------------------------

        if len(positions) < MAX_POSITIONS:

            candidates = []

            for symbol, df in data.items():

                if symbol in positions:
                    continue

                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)

                # Need previous completed day.
                if idx < 1:
                    continue

                previous = df.iloc[idx - 1]
                today = df.iloc[idx]

                # Signal generated from previous close.
                if not bool(previous["SIGNAL"]):
                    continue

                entry_price = float(today["Open"])

                if not np.isfinite(entry_price):
                    continue

                if entry_price <= 0:
                    continue

                atr = float(previous["ATR"])

                if not np.isfinite(atr):
                    continue

                stop_price = (
                    entry_price
                    - ATR_MULTIPLIER * atr
                )

                if stop_price <= 0:
                    continue

                stop_distance = (
                    entry_price - stop_price
                )

                # Risk-based position sizing.
                risk_amount = (
                    portfolio_value
                    * RISK_PER_TRADE
                )

                shares_by_risk = (
                    risk_amount
                    / stop_distance
                )

                # Maximum 20% portfolio allocation.
                max_position_value = (
                    portfolio_value
                    * MAX_POSITION_PCT
                )

                shares_by_position = (
                    max_position_value
                    / entry_price
                )

                shares = math.floor(
                    min(
                        shares_by_risk,
                        shares_by_position
                    )
                )

                if shares <= 0:
                    continue

                actual_entry = (
                    entry_price
                    * (1 + SLIPPAGE)
                )

                gross_cost = (
                    shares * actual_entry
                )

                commission = (
                    gross_cost * COMMISSION
                )

                total_cost = (
                    gross_cost + commission
                )

                # Maximum 70% invested.
                max_total_investment = (
                    portfolio_value
                    * MAX_INVESTED_PCT
                )

                current_invested = 0.0

                for p_symbol, p in positions.items():

                    p_df = data[p_symbol]

                    if date in p_df.index:

                        p_close = p_df.loc[
                            date,
                            "Close"
                        ]

                        if isinstance(
                            p_close,
                            pd.Series
                        ):
                            p_close = p_close.iloc[0]

                        current_invested += (
                            p["shares"]
                            * float(p_close)
                        )

                remaining_capacity = (
                    max_total_investment
                    - current_invested
                )

                if remaining_capacity <= 0:
                    continue

                if total_cost > remaining_capacity:

                    shares = math.floor(
                        remaining_capacity
                        / actual_entry
                    )

                    if shares <= 0:
                        continue

                    gross_cost = (
                        shares * actual_entry
                    )

                    commission = (
                        gross_cost * COMMISSION
                    )

                    total_cost = (
                        gross_cost + commission
                    )

                if total_cost > cash:
                    continue

                candidates.append({
                    "symbol": symbol,
                    "shares": shares,
                    "entry_price": actual_entry,
                    "stop_price": stop_price,
                    "cost": total_cost,
                    "score": previous["SIGNAL_SCORE"]
                })

            # Highest signal score first.
            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            available_slots = (
                MAX_POSITIONS
                - len(positions)
            )

            for candidate in candidates[
                :available_slots
            ]:

                if candidate["cost"] > cash:
                    continue

                symbol = candidate["symbol"]

                positions[symbol] = {
                    "shares": candidate["shares"],
                    "entry_price": candidate["entry_price"],
                    "stop_price": candidate["stop_price"],
                    "cost": candidate["cost"],
                    "entry_date": date
                }

                cash -= candidate["cost"]

                trades.append({
                    "date": date,
                    "symbol": symbol,
                    "side": "BUY",
                    "price": candidate["entry_price"],
                    "shares": candidate["shares"],
                    "value": candidate["cost"],
                    "pnl": 0.0,
                    "reason": "ENTRY"
                })


        # ---------------------------------
        # 4. END-OF-DAY EQUITY
        # ---------------------------------

        invested_value = 0.0

        for symbol, pos in positions.items():

            df = data[symbol]

            if date not in df.index:
                continue

            close_price = df.loc[
                date,
                "Close"
            ]

            if isinstance(
                close_price,
                pd.Series
            ):
                close_price = close_price.iloc[0]

            invested_value += (
                pos["shares"]
                * float(close_price)
            )

        portfolio_value = (
            cash + invested_value
        )

        equity_records.append({
            "date": date,
            "equity": portfolio_value,
            "cash": cash,
            "invested": invested_value,
            "positions": len(positions)
        })


    # ---------------------------------
    # 5. CLOSE REMAINING POSITIONS
    # ---------------------------------

    if all_dates:

        final_date = all_dates[-1]

        for symbol, pos in list(
            positions.items()
        ):

            df = data[symbol]

            if final_date not in df.index:
                continue

            close_price = df.loc[
                final_date,
                "Close"
            ]

            if isinstance(
                close_price,
                pd.Series
            ):
                close_price = close_price.iloc[0]

            exit_price = (
                float(close_price)
                * (1 - SLIPPAGE)
            )

            gross_value = (
                pos["shares"]
                * exit_price
            )

            commission = (
                gross_value
                * COMMISSION
            )

            net_value = (
                gross_value
                - commission
            )

            cash += net_value

            pnl = (
                net_value
                - pos["cost"]
            )

            trades.append({
                "date": final_date,
                "symbol": symbol,
                "side": "SELL",
                "price": exit_price,
                "shares": pos["shares"],
                "value": net_value,
                "pnl": pnl,
                "reason": "FINAL_CLOSE"
            })

        portfolio_value = cash

    return (
        pd.DataFrame(equity_records),
        pd.DataFrame(trades),
        portfolio_value
    )


# =========================
# PERFORMANCE
# =========================

def calculate_metrics(
    equity,
    trades,
    ending_capital
):

    if equity.empty:
        return {}

    equity = equity.copy()

    equity["date"] = pd.to_datetime(
        equity["date"]
    )

    equity = equity.sort_values(
        "date"
    )

    equity["daily_return"] = (
        equity["equity"]
        .pct_change()
        .fillna(0)
    )

    starting = START_CAPITAL

    total_return = (
        ending_capital / starting
        - 1
    )

    days = (
        equity["date"].iloc[-1]
        - equity["date"].iloc[0]
    ).days

    years = (
        days / 365.25
        if days > 0
        else 0
    )

    if years > 0:
        cagr = (
            ending_capital / starting
        ) ** (1 / years) - 1
    else:
        cagr = 0

    running_max = (
        equity["equity"]
        .cummax()
    )

    drawdown = (
        equity["equity"]
        / running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    daily_returns = (
        equity["daily_return"]
    )

    if daily_returns.std() != 0:

        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
            * np.sqrt(252)
        )

    else:
        sharpe = 0

    downside = daily_returns[
        daily_returns < 0
    ]

    if len(downside) > 0 and downside.std() != 0:

        sortino = (
            daily_returns.mean()
            / downside.std()
            * np.sqrt(252)
        )

    else:
        sortino = 0

    sells = trades[
        trades["side"] == "SELL"
    ].copy()

    if sells.empty:

        win_rate = 0
        profit_factor = 0
        avg_trade = 0
        avg_win = 0
        avg_loss = 0

    else:

        wins = sells[
            sells["pnl"] > 0
        ]

        losses = sells[
            sells["pnl"] < 0
        ]

        win_rate = (
            len(wins)
            / len(sells)
        )

        gross_profit = (
            wins["pnl"].sum()
        )

        gross_loss = abs(
            losses["pnl"].sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = np.inf

        avg_trade = sells[
            "pnl"
        ].mean()

        avg_win = (
            wins["pnl"].mean()
            if not wins.empty
            else 0
        )

        avg_loss = (
            losses["pnl"].mean()
            if not losses.empty
            else 0
        )

    avg_daily_pnl = (
        equity["equity"].diff().mean()
    )

    avg_daily_return = (
        daily_returns.mean()
    )

    return {
        "starting_capital": starting,
        "ending_capital": ending_capital,
        "total_return": total_return,
        "CAGR": cagr,
        "max_drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trade_count": len(sells),
        "avg_trade_pnl": avg_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_daily_pnl": avg_daily_pnl,
        "avg_daily_return": avg_daily_return,
        "data_start": equity["date"].iloc[0],
        "data_end": equity["date"].iloc[-1],
        "symbols": len(data)
    }


# =========================
# MAIN
# =========================

def main():

    os.makedirs(
        "results",
        exist_ok=True
    )

    data = download_data()

    if not data:

        raise RuntimeError(
            "No market data could be downloaded."
        )

    equity, trades, ending_capital = (
        run_backtest(data)
    )

    metrics = calculate_metrics(
        equity,
        trades,
        ending_capital
    )

    equity.to_csv(
        "results/equity_curve.csv",
        index=False
    )

    trades.to_csv(
        "results/trades.csv",
        index=False
    )

    summary = pd.DataFrame(
        [metrics]
    )

    summary.to_csv(
        "results/summary.csv",
        index=False
    )

    print("\n")
    print("=" * 60)
    print("BIST BACKTEST RESULTS")
    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, float):

            if key in [
                "total_return",
                "CAGR",
                "max_drawdown",
                "win_rate",
                "avg_daily_return"
            ]:

                print(
                    f"{key}: {value:.2%}"
                )

            else:

                print(
                    f"{key}: {value:,.2f}"
                )

        else:

            print(
                f"{key}: {value}"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
