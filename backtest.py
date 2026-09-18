import os
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS - AGGRESSIVE V2
# ============================================================

START_CAPITAL = 50_000.0

# Daha agresif risk
RISK_PER_TRADE = 0.0100

# Daha fazla eşzamanlı pozisyon
MAX_POSITIONS = 7

# Tek hisse maksimum ağırlığı
MAX_POSITION_PCT = 0.25

# Aktif sermaye
MAX_INVESTED_PCT = 0.85

# İşlem maliyetleri
COMMISSION = 0.001
SLIPPAGE = 0.001

# İlk stop
ATR_MULTIPLIER = 2.0

# Trailing stop
TRAILING_ATR_MULTIPLIER = 2.5

# Günlük zarar limiti
DAILY_LOSS_LIMIT = 0.03

# Portföy zirvesinden maksimum drawdown
MAX_DRAWDOWN_STOP = 0.15

# Piyasa breadth filtresi
MIN_BREADTH = 0.45

START_DATE = "2019-01-01"


# ============================================================
# SYMBOLS
# ============================================================

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
        [
            high_low,
            high_close,
            low_close
        ],
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

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):

        return pd.DataFrame()

    df = df[
        required_columns
    ].copy()

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["Close"]
        .ewm(
            span=200,
            adjust=False
        )
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

    df["RETURN_5"] = (
        df["Close"]
        / df["Close"].shift(5)
        - 1
    )

    # ========================================================
    # SIGNAL COMPONENTS
    # ========================================================

    df["C1_TREND"] = (
        df["EMA20"]
        > df["EMA50"]
    )

    df["C2_STRONG_TREND"] = (
        df["EMA50"]
        > df["EMA200"]
    )

    df["C3_RSI"] = (
        (df["RSI14"] >= 50)
        &
        (df["RSI14"] <= 72)
    )

    df["C4_VOLUME"] = (
        df["Volume"]
        >
        1.2 * df["VOL_MA20"]
    )

    df["C5_BREAKOUT"] = (
        df["Close"]
        >
        df["PREV_20_HIGH"]
    )

    df["C6_MOMENTUM"] = (
        df["RETURN_20"]
        > 0.05
    )

    df["C7_SHORT_MOMENTUM"] = (
        df["RETURN_5"]
        > 0.02
    )

    # ========================================================
    # SCORE
    # ========================================================

    df["SIGNAL_SCORE"] = (

        df["C1_TREND"].astype(int)

        +

        df["C2_STRONG_TREND"].astype(int)

        +

        df["C3_RSI"].astype(int)

        +

        df["C4_VOLUME"].astype(int)

        +

        df["C5_BREAKOUT"].astype(int)

        +

        df["C6_MOMENTUM"].astype(int)

        +

        df["C7_SHORT_MOMENTUM"].astype(int)
    )

    # Daha kaliteli sinyal
    df["SIGNAL"] = (
        df["SIGNAL_SCORE"]
        >= 5
    )

    return df


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    data = {}

    print("=" * 70)
    print("DOWNLOADING BIST DATA - AGGRESSIVE V2")
    print("=" * 70)

    for symbol in SYMBOLS:

        ticker = f"{symbol}.IS"

        print(
            f"Downloading {ticker} ..."
        )

        try:

            df = yf.download(
                ticker,
                start=START_DATE,
                auto_adjust=False,
                progress=False
            )

            if (
                df is None
                or df.empty
            ):

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
# PORTFOLIO VALUE
# ============================================================

def calculate_equity(
    data,
    positions,
    cash,
    date
):

    equity = cash

    for symbol, position in (
        positions.items()
    ):

        df = data[symbol]

        if date not in df.index:
            continue

        price = float(
            df.loc[date]["Close"]
        )

        if np.isfinite(price):

            equity += (
                position["shares"]
                * price
            )

    return equity


# ============================================================
# MARKET BREADTH
# ============================================================

def calculate_breadth(
    data,
    date
):

    values = []

    for symbol, df in data.items():

        if date not in df.index:
            continue

        row = df.loc[date]

        close = float(
            row["Close"]
        )

        ema50 = float(
            row["EMA50"]
        )

        if not np.isfinite(close):
            continue

        if not np.isfinite(ema50):
            continue

        values.append(
            close > ema50
        )

    if not values:

        return 0.0

    return (
        sum(values)
        / len(values)
    )


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

    peak_equity = START_CAPITAL

    trading_halted = False

    # ========================================================
    # DAILY LOOP
    # ========================================================

    for date in all_dates:

        # ====================================================
        # START OF DAY EQUITY
        # ====================================================

        day_start_equity = calculate_equity(
            data,
            positions,
            cash,
            date
        )

        if not np.isfinite(
            day_start_equity
        ):

            day_start_equity = cash

        # ====================================================
        # PEAK / DRAWDOWN CONTROL
        # ====================================================

        peak_equity = max(
            peak_equity,
            day_start_equity
        )

        drawdown = (
            day_start_equity
            / peak_equity
            - 1
        )

        if drawdown <= -MAX_DRAWDOWN_STOP:

            trading_halted = True

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

            if not np.isfinite(
                close_price
            ):

                continue

            entry_price = position[
                "entry_price"
            ]

            shares = position[
                "shares"
            ]

            stop_price = position[
                "stop_price"
            ]

            highest_close = position.get(
                "highest_close",
                entry_price
            )

            # =================================================
            # UPDATE HIGH
            # =================================================

            if close_price > highest_close:

                highest_close = close_price

                position[
                    "highest_close"
                ] = highest_close

                atr = float(
                    row["ATR14"]
                )

                if (
                    np.isfinite(atr)
                    and atr > 0
                ):

                    trailing_stop = (
                        highest_close
                        -
                        TRAILING_ATR_MULTIPLIER
                        * atr
                    )

                    position[
                        "stop_price"
                    ] = max(
                        stop_price,
                        trailing_stop
                    )

                    stop_price = position[
                        "stop_price"
                    ]

            # =================================================
            # EXIT CONDITIONS
            # =================================================

            stop_hit = (
                close_price
                <= stop_price
            )

            trend_break = (

                np.isfinite(
                    row["EMA20"]
                )

                and

                np.isfinite(
                    row["EMA50"]
                )

                and

                float(row["EMA20"])
                <
                float(row["EMA50"])
            )

            momentum_failure = (

                np.isfinite(
                    row["RSI14"]
                )

                and

                float(row["RSI14"])
                < 45
            )

            exit_signal = (

                stop_hit
                or
                trend_break
                or
                momentum_failure
            )

            if not exit_signal:
                continue

            # =================================================
            # EXIT
            # =================================================

            exit_price = (
                close_price
                * (1 - SLIPPAGE)
            )

            gross_value = (
                shares
                * exit_price
            )

            exit_commission = (
                gross_value
                * COMMISSION
            )

            cash += (
                gross_value
                -
                exit_commission
            )

            entry_cost = (
                entry_price
                * shares
            )

            entry_commission = (
                entry_cost
                * COMMISSION
            )

            gross_pnl = (
                exit_price
                -
                entry_price
            ) * shares

            net_pnl = (
                gross_pnl
                -
                exit_commission
                -
                entry_commission
            )

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

        # ====================================================
        # REMOVE CLOSED POSITIONS
        # ====================================================

        for symbol in symbols_to_close:

            del positions[symbol]

        # ====================================================
        # CURRENT EQUITY
        # ====================================================

        current_equity = calculate_equity(
            data,
            positions,
            cash,
            date
        )

        # ====================================================
        # DAILY LOSS
        # ====================================================

        daily_loss_pct = (

            current_equity
            /
            day_start_equity
            - 1

            if day_start_equity > 0

            else 0
        )

        # ====================================================
        # MARKET BREADTH
        # ====================================================

        breadth = calculate_breadth(
            data,
            date
        )

        # ====================================================
        # ENTRY PERMISSION
        # ====================================================

        allow_new_entries = (

            not trading_halted

            and

            daily_loss_pct
            >
            -DAILY_LOSS_LIMIT

            and

            breadth
            >=
            MIN_BREADTH
        )

        # ====================================================
        # FIND NEW ENTRIES
        # ====================================================

        if (
            allow_new_entries
            and
            len(positions)
            <
            MAX_POSITIONS
        ):

            portfolio_value = calculate_equity(
                data,
                positions,
                cash,
                date
            )

            invested_value = 0.0

            for symbol, position in (
                positions.items()
            ):

                df = data[symbol]

                if date not in df.index:
                    continue

                price = float(
                    df.loc[date]["Close"]
                )

                if np.isfinite(price):

                    invested_value += (
                        position["shares"]
                        * price
                    )

            candidates = []

            # =================================================
            # SCAN STOCKS
            # =================================================

            for symbol, df in data.items():

                if symbol in positions:
                    continue

                if date not in df.index:
                    continue

                idx = df.index.get_loc(
                    date
                )

                if idx == 0:
                    continue

                previous = df.iloc[
                    idx - 1
                ]

                current = df.iloc[
                    idx
                ]

                if not bool(
                    previous["SIGNAL"]
                ):

                    continue

                open_price = float(
                    current["Open"]
                )

                atr = float(
                    previous["ATR14"]
                )

                score = float(
                    previous[
                        "SIGNAL_SCORE"
                    ]
                )

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

                # =================================================
                # ENTRY
                # =================================================

                entry_price = (
                    open_price
                    * (1 + SLIPPAGE)
                )

                # =================================================
                # INITIAL STOP
                # =================================================

                stop_price = (
                    entry_price
                    -
                    ATR_MULTIPLIER
                    * atr
                )

                stop_distance = (
                    entry_price
                    -
                    stop_price
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
                        "score": score
                    }
                )

            # =================================================
            # BEST SIGNALS FIRST
            # =================================================

            candidates.sort(
                key=lambda x:
                x["score"],
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

                # =================================================
                # RISK
                # =================================================

                risk_amount = (
                    portfolio_value
                    *
                    RISK_PER_TRADE
                )

                shares_by_risk = (
                    risk_amount
                    /
                    stop_distance
                )

                # =================================================
                # MAX POSITION
                # =================================================

                max_position_value = (
                    portfolio_value
                    *
                    MAX_POSITION_PCT
                )

                shares_by_cap = (
                    max_position_value
                    /
                    entry_price
                )

                # =================================================
                # MAX ACTIVE CAPITAL
                # =================================================

                remaining_capital = max(
                    0,

                    portfolio_value
                    *
                    MAX_INVESTED_PCT
                    -
                    invested_value
                )

                shares_by_capital = (
                    remaining_capital
                    /
                    entry_price
                )

                values = [
                    shares_by_risk,
                    shares_by_cap,
                    shares_by_capital
                ]

                if not all(
                    np.isfinite(x)
                    for x in values
                ):

                    continue

                max_shares = min(values)

                if max_shares <= 0:
                    continue

                shares = math.floor(
                    max_shares
                )

                if shares <= 0:
                    continue

                # =================================================
                # COST
                # =================================================

                cost = (
                    shares
                    *
                    entry_price
                )

                commission = (
                    cost
                    *
                    COMMISSION
                )

                total_cost = (
                    cost
                    +
                    commission
                )

                if total_cost > cash:
                    continue

                # =================================================
                # OPEN
                # =================================================

                cash -= total_cost

                positions[symbol] = {

                    "entry_date": date,

                    "entry_price":
                        entry_price,

                    "shares":
                        shares,

                    "stop_price":
                        stop_price,

                    "highest_close":
                        entry_price
                }

                invested_value += cost

        # ====================================================
        # END OF DAY EQUITY
        # ====================================================

        equity = calculate_equity(
            data,
            positions,
            cash,
            date
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
                ),
                "breadth": breadth,
                "trading_halted":
                    trading_halted
            }
        )

    # ========================================================
    # FINAL CLOSE
    # ========================================================

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
            *
            (1 - SLIPPAGE)
        )

        shares = position[
            "shares"
        ]

        gross_value = (
            shares
            *
            exit_price
        )

        commission = (
            gross_value
            *
            COMMISSION
        )

        cash += (
            gross_value
            -
            commission
        )

        entry_cost = (
            position["entry_price"]
            *
            shares
        )

        entry_commission = (
            entry_cost
            *
            COMMISSION
        )

        gross_pnl = (
            exit_price
            -
            position["entry_price"]
        ) * shares

        net_pnl = (
            gross_pnl
            -
            commission
            -
            entry_commission
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
                "exit_reason":
                    "FINAL_CLOSE"
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
    # PERFORMANCE
    # ========================================================

    ending_capital = cash

    total_return = (
        ending_capital
        /
        START_CAPITAL
        - 1
    )

    days = max(
        1,
        (
            pd.Timestamp(
                all_dates[-1]
            )
            -
            pd.Timestamp(
                all_dates[0]
            )
        ).days
    )

    years = (
        days
        /
        365.25
    )

    if (
        years > 0
        and
        ending_capital > 0
    ):

        cagr = (
            (
                ending_capital
                /
                START_CAPITAL
            )
            **
            (1 / years)
            -
            1
        )

    else:

        cagr = 0

    # ========================================================
    # DAILY RETURNS
    # ========================================================

    equity_df[
        "daily_return"
    ] = (
        equity_df["equity"]
        .pct_change()
        .fillna(0)
    )

    equity_df[
        "peak"
    ] = (
        equity_df["equity"]
        .cummax()
    )

    equity_df[
        "drawdown"
    ] = (
        equity_df["equity"]
        /
        equity_df["peak"]
        -
        1
    )

    max_drawdown = (
        equity_df[
            "drawdown"
        ].min()
    )

    daily_returns = (
        equity_df[
            "daily_return"
        ]
    )

    # ========================================================
    # SHARPE
    # ========================================================

    if (
        len(daily_returns) > 1
        and
        daily_returns.std() > 0
    ):

        sharpe = (
            daily_returns.mean()
            /
            daily_returns.std()
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
        and
        downside.std() > 0
    ):

        sortino = (
            daily_returns.mean()
            /
            downside.std()
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
            /
            len(trades_df)
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
                /
                gross_loss
            )

        else:

            profit_factor = np.inf

        expectancy = (
            trades_df["pnl"]
            .mean()
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
        equity_df[
            "daily_return"
        ].mean()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "start_capital":
                    START_CAPITAL,

                "ending_capital":
                    ending_capital,

                "total_return":
                    total_return,

                "CAGR":
                    cagr,

                "max_drawdown":
                    max_drawdown,

                "Sharpe":
                    sharpe,

                "Sortino":
                    sortino,

                "win_rate":
                    win_rate,

                "profit_factor":
                    profit_factor,

                "expectancy_TL":
                    expectancy,

                "average_win_TL":
                    avg_win,

                "average_loss_TL":
                    avg_loss,

                "average_daily_pnl_TL":
                    average_daily_pnl,

                "average_daily_return":
                    average_daily_return,

                "trade_count":
                    len(trades_df),

                "data_start":
                    str(all_dates[0]),

                "data_end":
                    str(all_dates[-1]),

                "symbols_requested":
                    len(SYMBOLS),

                "symbols_downloaded":
                    len(data),

                "commission":
                    COMMISSION,

                "slippage":
                    SLIPPAGE,

                "risk_per_trade":
                    RISK_PER_TRADE,

                "max_positions":
                    MAX_POSITIONS,

                "max_position_pct":
                    MAX_POSITION_PCT,

                "max_invested_pct":
                    MAX_INVESTED_PCT,

                "daily_loss_limit":
                    DAILY_LOSS_LIMIT,

                "max_drawdown_stop":
                    MAX_DRAWDOWN_STOP,

                "trailing_atr_multiplier":
                    TRAILING_ATR_MULTIPLIER,

                "min_breadth":
                    MIN_BREADTH
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
    # SAVE
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
    # PRINT
    # ========================================================

    print()

    print("=" * 70)
    print("BIST BACKTEST RESULTS - AGGRESSIVE V2")
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
        f"{result['data_start']} -> "
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

    print()

    print("V2 RISK SETTINGS")
    print("-" * 70)

    print(
        f"Risk / trade     : "
        f"{RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"Max positions    : "
        f"{MAX_POSITIONS}"
    )

    print(
        f"Max position     : "
        f"{MAX_POSITION_PCT * 100:.0f}%"
    )

    print(
        f"Active capital   : "
        f"{MAX_INVESTED_PCT * 100:.0f}%"
    )

    print(
        f"Daily loss stop  : "
        f"{DAILY_LOSS_LIMIT * 100:.1f}%"
    )

    print(
        f"DD stop          : "
        f"{MAX_DRAWDOWN_STOP * 100:.1f}%"
    )

    print(
        f"Min breadth      : "
        f"{MIN_BREADTH * 100:.0f}%"
    )

    print(
        f"Trailing ATR     : "
        f"{TRAILING_ATR_MULTIPLIER:.1f}x"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
