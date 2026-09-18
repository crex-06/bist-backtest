import os
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================

START_CAPITAL = 50_000.0

RISK_PER_TRADE = 0.0075
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.20
MAX_INVESTED_PCT = 0.70

COMMISSION = 0.001
SLIPPAGE = 0.001

ATR_MULTIPLIER = 2.0

START_DATE = "2019-01-01"

# KOZAA ve KOZAL Yahoo Finance tarafında bulunamadığı için
# geçici olarak veri evreninden çıkarıldı.
SYMBOLS = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR",
    "BIMAS", "BRSAN", "CCOLA", "DOAS", "DOHOL",
    "ECILC", "EKGYO", "ENKAI", "EREGL", "FENER",
    "FROTO", "GARAN", "GUBRF", "HALKB", "HEKTS",
    "ISCTR", "KCHOL", "KONTR", "KRDMD", "MGROS",
    "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA",
    "SISE", "SKBNK", "TAVHL", "TCELL", "THYAO",
    "TKFEN", "TOASO", "TUPRS", "ULKER", "VAKBN",
    "YKBNK"
]


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):

    df = df.copy()

    if df.empty:
        return df

    # Yahoo bazen MultiIndex döndürebiliyor.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    # Eksik kolon varsa bu sembolü kullanma.
    if not all(
        column in df.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    df = df[required_columns].copy()

    # --------------------------------------------------------
    # TECHNICAL INDICATORS
    # --------------------------------------------------------

    df["EMA20"] = (
        df["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    df["ATR14"] = calculate_atr(
        df,
        14
    )

    df["VOL_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["PREV_20_HIGH"] = (
        df["High"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["RETURN_20"] = (
        df["Close"]
        / df["Close"].shift(20)
        - 1
    )

    # --------------------------------------------------------
    # SIGNAL COMPONENTS
    # --------------------------------------------------------

    df["C1_TREND"] = (
        df["EMA20"] > df["EMA50"]
    )

    df["C2_RSI"] = (
        (df["RSI14"] >= 50)
        & (df["RSI14"] <= 72)
    )

    df["C3_VOLUME"] = (
        df["Volume"]
        > 1.2 * df["VOL_MA20"]
    )

    df["C4_BREAKOUT"] = (
        df["Close"]
        > df["PREV_20_HIGH"]
    )

    df["C5_MOMENTUM"] = (
        df["RETURN_20"] > 0.05
    )

    df["SIGNAL_SCORE"] = (
        df["C1_TREND"].astype(int)
        + df["C2_RSI"].astype(int)
        + df["C3_VOLUME"].astype(int)
        + df["C4_BREAKOUT"].astype(int)
        + df["C5_MOMENTUM"].astype(int)
    )

    df["SIGNAL"] = (
        df["SIGNAL_SCORE"] >= 4
    )

    return df


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    data = {}

    print("=" * 70)
    print("DOWNLOADING BIST DATA")
    print("=" * 70)

    for symbol in SYMBOLS:

        ticker = f"{symbol}.IS"

        print(f"Downloading {ticker} ...")

        try:

            df = yf.download(
                ticker,
                start=START_DATE,
                auto_adjust=False,
                progress=False
            )

            if df is None or df.empty:

                print(
                    f"  No data: {symbol}"
                )

                continue

            df = prepare_data(df)

            if not df.empty:

                data[symbol] = df

                print(
                    f"  OK: {len(df)} rows"
                )

            else:

                print(
                    f"  Invalid data: {symbol}"
                )

        except Exception as e:

            print(
                f"  ERROR {symbol}: {e}"
            )

    print()

    print(
        f"Downloaded {len(data)} / "
        f"{len(SYMBOLS)} symbols"
    )

    return data


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(data):

    all_dates = sorted(
        set(
            date
            for df in data.values()
            for date in df.index
        )
    )

    if not all_dates:

        raise RuntimeError(
            "No market data available."
        )

    cash = START_CAPITAL

    positions = {}

    equity_records = []

    trades = []

    cumulative_realized_pnl = 0.0

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in all_dates:

        # ====================================================
        # EXIT EXISTING POSITIONS
        # ====================================================

        symbols_to_close = []

        for symbol, position in list(
            positions.items()
        ):

            df = data[symbol]

            if date not in df.index:
                continue

            row = df.loc[date]

            close_price = float(
                row["Close"]
            )

            entry_price = position[
                "entry_price"
            ]

            shares = position[
                "shares"
            ]

            stop_price = position[
                "stop_price"
            ]

            # ------------------------------------------------
            # SAFETY CHECK
            # ------------------------------------------------

            if not np.isfinite(close_price):
                continue

            if not np.isfinite(entry_price):
                continue

            if not np.isfinite(stop_price):
                continue

            # ------------------------------------------------
            # EXIT CONDITIONS
            # ------------------------------------------------

            stop_hit = (
                close_price <= stop_price
            )

            trend_break = (
                np.isfinite(row["EMA20"])
                and np.isfinite(row["EMA50"])
                and float(row["EMA20"])
                < float(row["EMA50"])
            )

            momentum_failure = (
                np.isfinite(row["RSI14"])
                and float(row["RSI14"]) < 45
            )

            exit_signal = (
                stop_hit
                or trend_break
                or momentum_failure
            )

            if exit_signal:

                exit_price = (
                    close_price
                    * (1 - SLIPPAGE)
                )

                gross_value = (
                    shares
                    * exit_price
                )

                commission = (
                    gross_value
                    * COMMISSION
                )

                cash += (
                    gross_value
                    - commission
                )

                gross_pnl = (
                    exit_price
                    - entry_price
                ) * shares

                entry_cost = (
                    entry_price
                    * shares
                )

                entry_commission = (
                    entry_cost
                    * COMMISSION
                )

                net_pnl = (
                    gross_pnl
                    - commission
                    - entry_commission
                )

                cumulative_realized_pnl += net_pnl

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": position[
                            "entry_date"
                        ],
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares,
                        "stop_price": stop_price,
                        "pnl": net_pnl,
                        "return_pct": (
                            net_pnl / entry_cost
                            if entry_cost > 0
                            else 0
                        ),
                        "exit_reason": (
                            "STOP"
                            if stop_hit
                            else "TREND/MOMENTUM"
                        )
                    }
                )

                symbols_to_close.append(
                    symbol
                )

        for symbol in symbols_to_close:

            del positions[symbol]

        # ====================================================
        # FIND NEW ENTRIES
        # ====================================================

        if len(positions) < MAX_POSITIONS:

            # ------------------------------------------------
            # CURRENT PORTFOLIO VALUE
            # ------------------------------------------------

            portfolio_value = cash

            for symbol, position in (
                positions.items()
            ):

                df = data[symbol]

                if date in df.index:

                    close_price = float(
                        df.loc[date]["Close"]
                    )

                    if np.isfinite(close_price):

                        portfolio_value += (
                            position["shares"]
                            * close_price
                        )

            # ------------------------------------------------
            # INVESTED VALUE
            # ------------------------------------------------

            invested_value = 0.0

            for symbol, position in (
                positions.items()
            ):

                df = data[symbol]

                if date in df.index:

                    close_price = float(
                        df.loc[date]["Close"]
                    )

                    if np.isfinite(close_price):

                        invested_value += (
                            position["shares"]
                            * close_price
                        )

            # ------------------------------------------------
            # PORTFOLIO SAFETY CHECK
            # ------------------------------------------------

            if (
                not np.isfinite(
                    portfolio_value
                )
                or portfolio_value <= 0
            ):

                continue

            # ------------------------------------------------
            # CANDIDATES
            # ------------------------------------------------

            candidates = []

            for symbol, df in data.items():

                if symbol in positions:
                    continue

                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)

                if idx == 0:
                    continue

                prev_row = df.iloc[
                    idx - 1
                ]

                current_row = df.iloc[
                    idx
                ]

                # --------------------------------------------
                # PREVIOUS DAY SIGNAL
                # --------------------------------------------

                if not bool(
                    prev_row["SIGNAL"]
                ):

                    continue

                open_price = float(
                    current_row["Open"]
                )

                atr = float(
                    prev_row["ATR14"]
                )

                # --------------------------------------------
                # CRITICAL NaN / INF CHECK
                # --------------------------------------------

                if not np.isfinite(
                    open_price
                ):

                    continue

                if open_price <= 0:
                    continue

                if not np.isfinite(atr):
                    continue

                if atr <= 0:
                    continue

                # --------------------------------------------
                # ENTRY / STOP
                # --------------------------------------------

                entry_price = (
                    open_price
                    * (1 + SLIPPAGE)
                )

                stop_price = (
                    entry_price
                    - ATR_MULTIPLIER * atr
                )

                stop_distance = (
                    entry_price
                    - stop_price
                )

                if not np.isfinite(
                    entry_price
                ):
                    continue

                if not np.isfinite(
                    stop_price
                ):
                    continue

                if not np.isfinite(
                    stop_distance
                ):
                    continue

                if stop_distance <= 0:
                    continue

                candidates.append(
                    {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "stop_distance": stop_distance,
                        "score": float(
                            prev_row[
                                "SIGNAL_SCORE"
                            ]
                        )
                    }
                )

            # ------------------------------------------------
            # HIGHEST SCORE FIRST
            # ------------------------------------------------

            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            # =================================================
            # OPEN POSITIONS
            # =================================================

            for candidate in candidates:

                if (
                    len(positions)
                    >= MAX_POSITIONS
                ):

                    break

                symbol = candidate[
                    "symbol"
                ]

                entry_price = candidate[
                    "entry_price"
                ]

                stop_price = candidate[
                    "stop_price"
                ]

                stop_distance = candidate[
                    "stop_distance"
                ]

                # --------------------------------------------
                # FINAL SAFETY CHECK
                # --------------------------------------------

                if not all(
                    np.isfinite(x)
                    for x in [
                        entry_price,
                        stop_price,
                        stop_distance,
                        portfolio_value,
                        cash,
                        invested_value
                    ]
                ):

                    continue

                if (
                    entry_price <= 0
                    or stop_distance <= 0
                    or portfolio_value <= 0
                ):

                    continue

                # --------------------------------------------
                # RISK AMOUNT
                # --------------------------------------------

                risk_amount = (
                    portfolio_value
                    * RISK_PER_TRADE
                )

                if not np.isfinite(
                    risk_amount
                ):

                    continue

                if risk_amount <= 0:
                    continue

                # --------------------------------------------
                # POSITION SIZE BY RISK
                # --------------------------------------------

                shares_by_risk = (
                    risk_amount
                    / stop_distance
                )

                # --------------------------------------------
                # POSITION SIZE BY MAX POSITION
                # --------------------------------------------

                max_position_value = (
                    portfolio_value
                    * MAX_POSITION_PCT
                )

                shares_by_cap = (
                    max_position_value
                    / entry_price
                )

                # --------------------------------------------
                # POSITION SIZE BY CAPITAL
                # --------------------------------------------

                remaining_capital = (
                    portfolio_value
                    * MAX_INVESTED_PCT
                    - invested_value
                )

                shares_by_capital = (
                    max(
                        0,
                        remaining_capital
                    )
                    / entry_price
                )

                # --------------------------------------------
                # CRITICAL NaN / INF PROTECTION
                # --------------------------------------------

                position_size_values = [
                    shares_by_risk,
                    shares_by_cap,
                    shares_by_capital
                ]

                if not all(
                    np.isfinite(x)
                    for x in position_size_values
                ):

                    continue

                max_shares = min(
                    position_size_values
                )

                if not np.isfinite(
                    max_shares
                ):

                    continue

                if max_shares <= 0:
                    continue

                # --------------------------------------------
                # INTEGER SHARE COUNT
                # --------------------------------------------

                shares = math.floor(
                    max_shares
                )

                if shares <= 0:
                    continue

                # --------------------------------------------
                # COST
                # --------------------------------------------

                cost = (
                    shares
                    * entry_price
                )

                commission = (
                    cost
                    * COMMISSION
                )

                total_cost = (
                    cost
                    + commission
                )

                if not all(
                    np.isfinite(x)
                    for x in [
                        cost,
                        commission,
                        total_cost
                    ]
                ):

                    continue

                if total_cost <= 0:
                    continue

                if total_cost > cash:
                    continue

                # --------------------------------------------
                # OPEN POSITION
                # --------------------------------------------

                cash -= total_cost

                positions[symbol] = {
                    "entry_date": date,
                    "entry_price": entry_price,
                    "shares": shares,
                    "stop_price": stop_price
                }

                invested_value += cost

        # ====================================================
        # DAILY EQUITY
        # ====================================================

        equity = cash

        for symbol, position in (
            positions.items()
        ):

            df = data[symbol]

            if date in df.index:

                close_price = float(
                    df.loc[date]["Close"]
                )

                if np.isfinite(
                    close_price
                ):

                    equity += (
                        position["shares"]
                        * close_price
                    )

        if not np.isfinite(equity):

            equity = cash

        equity_records.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "positions": len(
                    positions
                )
            }
        )

    # ========================================================
    # FINAL CLOSE
    # ========================================================

    if all_dates:

        final_date = all_dates[-1]

        for symbol, position in list(
            positions.items()
        ):

            df = data[symbol]

            if final_date not in df.index:
                continue

            final_close = float(
                df.loc[final_date]["Close"]
            )

            if not np.isfinite(
                final_close
            ):

                continue

            exit_price = (
                final_close
                * (1 - SLIPPAGE)
            )

            shares = position[
                "shares"
            ]

            gross_value = (
                shares
                * exit_price
            )

            commission = (
                gross_value
                * COMMISSION
            )

            cash += (
                gross_value
                - commission
            )

            gross_pnl = (
                exit_price
                - position["entry_price"]
            ) * shares

            entry_cost = (
                position["entry_price"]
                * shares
            )

            entry_commission = (
                entry_cost
                * COMMISSION
            )

            net_pnl = (
                gross_pnl
                - commission
                - entry_commission
            )

            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": position[
                        "entry_date"
                    ],
                    "exit_date": final_date,
                    "entry_price": position[
                        "entry_price"
                    ],
                    "exit_price": exit_price,
                    "shares": shares,
                    "stop_price": position[
                        "stop_price"
                    ],
                    "pnl": net_pnl,
                    "return_pct": (
                        net_pnl / entry_cost
                        if entry_cost > 0
                        else 0
                    ),
                    "exit_reason": "FINAL_CLOSE"
                }
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    equity_df = pd.DataFrame(
        equity_records
    )

    trades_df = pd.DataFrame(
        trades
    )

    if equity_df.empty:

        raise RuntimeError(
            "Equity curve is empty."
        )

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    ending_capital = cash

    total_return = (
        ending_capital
        / START_CAPITAL
        - 1
    )

    days = max(
        1,
        (
            pd.Timestamp(
                all_dates[-1]
            )
            - pd.Timestamp(
                all_dates[0]
            )
        ).days
    )

    years = days / 365.25

    if (
        years > 0
        and ending_capital > 0
    ):

        cagr = (
            (
                ending_capital
                / START_CAPITAL
            )
            ** (1 / years)
            - 1
        )

    else:

        cagr = 0

    # ========================================================
    # EQUITY METRICS
    # ========================================================

    equity_df[
        "daily_return"
    ] = (
        equity_df["equity"]
        .pct_change()
        .fillna(0)
    )

    equity_df["peak"] = (
        equity_df["equity"]
        .cummax()
    )

    equity_df["drawdown"] = (
        equity_df["equity"]
        / equity_df["peak"]
        - 1
    )

    max_drawdown = (
        equity_df["drawdown"]
        .min()
    )

    daily_returns = (
        equity_df["daily_return"]
    )

    # ========================================================
    # SHARPE
    # ========================================================

    if (
        len(daily_returns) > 1
        and daily_returns.std() > 0
    ):

        sharpe = (
            daily_returns.mean()
            / daily_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe = 0

    # ========================================================
    # SORTINO
    # ========================================================

    downside = (
        daily_returns[
            daily_returns < 0
        ]
    )

    if (
        len(downside) > 1
        and downside.std() > 0
    ):

        sortino = (
            daily_returns.mean()
            / downside.std()
        ) * np.sqrt(252)

    else:

        sortino = 0

    # ========================================================
    # TRADE METRICS
    # ========================================================

    if not trades_df.empty:

        wins = trades_df[
            trades_df["pnl"] > 0
        ]

        losses = trades_df[
            trades_df["pnl"] < 0
        ]

        win_rate = (
            len(wins)
            / len(trades_df)
        )

        gross_profit = (
            wins["pnl"].sum()
            if not wins.empty
            else 0
        )

        gross_loss = (
            abs(
                losses["pnl"].sum()
            )
            if not losses.empty
            else 0
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = np.inf

        expectancy = (
            trades_df["pnl"].mean()
        )

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

    else:

        win_rate = 0
        profit_factor = 0
        expectancy = 0
        avg_win = 0
        avg_loss = 0

    # ========================================================
    # DAILY METRICS
    # ========================================================

    average_daily_pnl = (
        equity_df["equity"]
        .diff()
        .mean()
    )

    average_daily_return = (
        equity_df["daily_return"]
        .mean()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "start_capital": START_CAPITAL,
                "ending_capital": ending_capital,
                "total_return": total_return,
                "CAGR": cagr,
                "max_drawdown": max_drawdown,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy_TL": expectancy,
                "average_win_TL": avg_win,
                "average_loss_TL": avg_loss,
                "average_daily_pnl_TL": average_daily_pnl,
                "average_daily_return": average_daily_return,
                "trade_count": len(trades_df),
                "data_start": str(
                    all_dates[0]
                ),
                "data_end": str(
                    all_dates[-1]
                ),
                "symbols_requested": len(
                    SYMBOLS
                ),
                "symbols_downloaded": len(
                    data
                ),
                "commission": COMMISSION,
                "slippage": SLIPPAGE
            }
        ]
    )

    return (
        equity_df,
        trades_df,
        summary
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        "results",
        exist_ok=True
    )

    data = download_data()

    if not data:

        raise RuntimeError(
            "No symbols were downloaded."
        )

    (
        equity_df,
        trades_df,
        summary
    ) = run_backtest(data)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    equity_df.to_csv(
        "results/equity_curve.csv",
        index=False
    )

    trades_df.to_csv(
        "results/trades.csv",
        index=False
    )

    summary.to_csv(
        "results/summary.csv",
        index=False
    )

    result = summary.iloc[0]

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()

    print("=" * 70)
    print("BIST BACKTEST RESULTS")
    print("=" * 70)

    print(
        f"Starting capital : "
        f"{result['start_capital']:,.2f} TL"
    )

    print(
        f"Ending capital   : "
        f"{result['ending_capital']:,.2f} TL"
    )

    print(
        f"Total return     : "
        f"{result['total_return'] * 100:.2f}%"
    )

    print(
        f"CAGR             : "
        f"{result['CAGR'] * 100:.2f}%"
    )

    print(
        f"Max drawdown     : "
        f"{result['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe           : "
        f"{result['Sharpe']:.2f}"
    )

    print(
        f"Sortino          : "
        f"{result['Sortino']:.2f}"
    )

    print(
        f"Win rate         : "
        f"{result['win_rate'] * 100:.2f}%"
    )

    print(
        f"Profit factor    : "
        f"{result['profit_factor']:.2f}"
    )

    print(
        f"Trade count      : "
        f"{int(result['trade_count'])}"
    )

    print(
        f"Avg daily P&L    : "
        f"{result['average_daily_pnl_TL']:,.2f} TL"
    )

    print(
        f"Avg daily return : "
        f"{result['average_daily_return'] * 100:.4f}%"
    )

    print(
        f"Data period      : "
        f"{result['data_start']} → "
        f"{result['data_end']}"
    )

    print(
        f"Symbols          : "
        f"{int(result['symbols_downloaded'])}/"
        f"{int(result['symbols_requested'])}"
    )

    print(
        f"Commission       : "
        f"{COMMISSION * 100:.2f}%"
    )

    print(
        f"Slippage         : "
        f"{SLIPPAGE * 100:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
