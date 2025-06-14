import pandas as pd
from datetime import datetime, timedelta
from utils.LabelsDict import tickers
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import List, Optional
from utils.logger import Logger
import os
from tqdm import tqdm
from pycbrf.toolbox import ExchangeRates

log = Logger(__name__).get_logger()


def download_finam_quotes(
    ticker: str, period: int, start: str, end: str, bucket_size: int = 100
) -> pd.DataFrame:
    """
    Downloads historical market data from Finam and returns as DataFrame

    Args:
        ticker: Instrument ticker (e.g. 'SBER')
        period: Timeframe from predefined values:
            2 - 1 min, 3 - 5 min, 4 - 10 min, 5 - 15 min,
            6 - 30 min, 7 - hour, 8 - day, 9 - week, 10 - month
        start: Start date in DD.MM.YYYY format
        end: End date in DD.MM.YYYY format
        bucket_size: Number of records per request (default: 100)

    Returns:
        pd.DataFrame: DataFrame containing historical market data with columns:
            '<TICKER>', '<PER>', '<DATE>', '<TIME>', '<OPEN>', '<HIGH>',
            '<LOW>', '<CLOSE>', '<VOL>'
    """

    if ticker not in tickers:
        raise ValueError(
            f"\nTicker {ticker} not found. Available tickers: {list(tickers.keys())}"
        )

    period_deltas = {
        2: timedelta(minutes=1),
        3: timedelta(minutes=5),
        4: timedelta(minutes=10),
        5: timedelta(minutes=15),
        6: timedelta(minutes=30),
        7: timedelta(hours=1),
        8: timedelta(days=1),
        9: timedelta(weeks=1),
        10: timedelta(days=30),
    }

    if period not in period_deltas:
        raise ValueError(
            f"Invalid period: {period}. Available periods: "
            f"2 (1min), 3 (5min), 4 (10min), 5 (15min), "
            f"6 (30min), 7 (hour), 8 (day), 9 (week), 10 (month)"
        )

    delta = period_deltas[period]
    current_start = datetime.strptime(start, "%d.%m.%Y")
    end_date = datetime.strptime(end, "%d.%m.%Y")
    result_df, first_bucket = None, True

    while current_start <= end_date:
        start_rev = current_start.strftime("%Y%m%d")
        end_rev = min(current_start + delta * bucket_size, end_date).strftime("%Y%m%d")

        params = urlencode(
            [
                ("market", 0),  # Market type (0 - stocks)
                ("em", tickers[ticker]),
                ("code", ticker),
                ("apply", 0),
                ("df", current_start.day),
                ("mf", current_start.month - 1),
                ("yf", current_start.year),
                ("from", current_start.date()),
                ("dt", end_date.day),
                ("mt", end_date.month - 1),
                ("yt", end_date.year),
                ("to", end_date.date()),
                ("p", period),
                ("f", f"{ticker}_{start_rev}_{end_rev}"),
                ("e", ".csv"),
                ("cn", ticker),
                ("dtf", 1),  # Date format (1 - YYYYMMDD)
                ("tmf", 1),  # Time format (1 - HHMMSS)
                ("MSOR", 0),  # Candle time (0 - open; 1 - close)
                ("mstime", "on"),  # Moscow time zone
                ("mstimever", 1),
                ("sep", 1),
                ("sep2", 1),
                ("datf", 1),
                ("at", 1 if first_bucket else 0),
            ]
        )

        url = f"http://export.finam.ru/{ticker}_{start_rev}_{end_rev}.csv?{params}"

        try:
            with urlopen(url) as response:
                content = response.read().decode("utf-8")
                lines = [line.split(",") for line in content.splitlines()]

                if not lines:
                    break

                if lines[0][0] == "Запрашиваемая вами глубина недоступна":
                    current_start += delta
                    continue

                if first_bucket:
                    result_df = pd.DataFrame(lines[1:], columns=lines[0])
                    first_bucket = False
                else:
                    result_df = pd.concat([result_df, pd.DataFrame(lines[1:])])

                last_date = datetime.strptime(
                    f"{lines[-1][2]} {lines[-1][3]}", "%Y%m%d %H%M%S"
                )
                current_start = last_date + delta

        except Exception as e:
            log.error(f"Download error for ticker {ticker}: {str(e)}")
            break

    log.info(
        f"Downloaded dates range for ticker {ticker} : "
        f"[{pd.to_datetime(result_df['<DATE>'].min())} : {pd.to_datetime(result_df['<DATE>'].max())}]"
    )

    return result_df


def load_stock_data(
    tickers_list: List[str],
    start_date: str,
    end_date: str,
    step: int,
    bucket_size: int = 10,
) -> Optional[pd.DataFrame]:
    """
    Loads historical stock data for multiple tickers from Finam and combines
    into a single DataFrame.

    Args:
        tickers_list: List of tickers to download (e.g. ['SBER', 'GAZP']).
        start_date: Start date in DD.MM.YYYY format.
        end_date: End date in DD.MM.YYYY format.
        step: Timeframe step from predefined values:
            2 - 1 min, 3 - 5 min, 4 - 10 min, 5 - 15 min,
            6 - 30 min, 7 - hour, 8 - day, 9 - week, 10 - month.
        bucket_size: Number of records per request (default: 10).
            Smaller values may be needed for intraday data to avoid request limits.

    Returns:
        pd.DataFrame: Combined DataFrame containing historical data for all
        requested tickers.
        Columns include:
            '<TICKER>', '<PER>', '<DATE>', '<TIME>', '<OPEN>', '<HIGH>',
            '<LOW>', '<CLOSE>', '<VOL>', 'datetime' (combined date+time).
        Returns None if no data was downloaded for any ticker.

    Raises:
        ValueError: If `tickers_list` is empty or invalid dates are provided.
    """
    if not tickers_list:
        raise ValueError("Tickers list cannot be empty")

    total_df: Optional[pd.DataFrame] = None

    for ticker in tickers_list:
        ticker_data = download_finam_quotes(
            ticker=ticker,
            period=step,
            start=start_date,
            end=end_date,
            bucket_size=bucket_size,
        )

        if total_df is None:
            total_df = ticker_data
        else:
            total_df = pd.concat([total_df, ticker_data])

    return total_df


def load_multipliers(companies_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads financial multipliers for a given list of companies from CSV files.

    Reads CSV files for each company in `companies_list`, filters predefined financial
    multipliers (P/E, P/FCF, etc.), and combines them into a single DataFrame.

    Args:
        companies_list: List of company tickers (e.g., ['GAZP', 'SBER']). If None,
                       uses a default list of Russian blue-chip companies.

    Returns:
        pd.DataFrame: A DataFrame containing the selected financial multipliers
                     for each company, with columns:
                     - 'company' (str): Company ticker.
                     - 'characteristic' (str): Financial multiplier (e.g., 'P/E').
                     - Year columns (2014–2024): Multiplier values for each year.
    """

    default_companies = [
        "GAZP",
        "LKOH",
        "ROSN",
        "SBER",
        "VTBR",
        "T",
        "GMKN",
        "NLMK",
        "RUAL",
        "MTSS",
        "RTKM",
        "MGNT",
        "X5",
        "LNTA",
    ]
    companies_list = default_companies if companies_list is None else companies_list
    multipliers = [
        "P/E",
        "P/FCF",
        "P/S",
        "P/BV",
        "EV/EBITDA",
        "Долг/EBITDA",
        "Капитализация, млрд руб",
        "Долг, млрд руб",
        "Чистый долг, млрд руб",
    ]

    macro = None
    for company in companies_list:
        tmp = pd.read_csv(f"data/multiplicators/{company}.csv", sep=";")
        tmp.columns = ["characteristic"] + list(tmp.columns)[1:]

        tmp = (
            tmp.assign(temp=lambda x: 1 * tmp["characteristic"].isin(multipliers))
            .query("temp == 1")
            .drop(columns="temp")
            .assign(company=company)
        )

        macro = tmp if macro is None else pd.concat([macro, tmp])

    macro = macro.rename(
        columns={
            f"{year}Q{q}": f"{year}_{q}"
            for year in range(2000, 2025)
            for q in range(1, 5)
        }
    )

    macro = macro[
        ["company", "characteristic"]
        + [
            f"{year}_{q}"
            for year in range(2014, 2025)
            for q in range(1, 5)
            if f"{year}_{q}" in macro.columns
        ]
    ]

    return macro


def get_rubusd_exchange_rate(
    dt_calc: str, dt_start: str, update_backup: bool = False, use_backup: bool = False
) -> pd.DataFrame:
    rubusd_df_path = f"data/macro/rubusd.csv"

    if use_backup:
        if not os.path.exists(rubusd_df_path):
            log.error(
                f"Backup file for usd/rub exchange rates not found. Please, update it."
            )
            return

        rates = pd.read_csv(rubusd_df_path)
        log.info(
            f"Exchange rates for usd/rub will be use from backup. Last actual date: {rates.date.max()}"
        )

        return rates

    if os.path.exists(rubusd_df_path):
        rates = pd.read_csv(rubusd_df_path)
        start_date = rates.date.max()
    else:
        rates = None
        start_date = dt_start

    date_range = pd.date_range(
        start=pd.to_datetime(start_date, format="%Y-%m-%d"),
        end=pd.to_datetime(dt_calc, format="%Y-%m-%d"),
        freq="D",
    )

    if start_date != dt_calc:
        rates_additional = []
        log.info(
            f"Downloading new usd/rub exchange rates from {start_date} to {dt_calc}"
        )
        for date in tqdm(date_range):
            rates_additional.append(
                (date.strftime("%Y-%m-%d"), float(ExchangeRates(date)["USD"].value))
            )

        rates = (
            pd.DataFrame(rates_additional, columns=["date", "rubusd_exchange_rate"])
            if rates is None
            else pd.concat(
                [rates, pd.DataFrame(rates_additional, columns=rates.columns)]
            )
        )

    if update_backup:
        rates.to_csv(rubusd_df_path, index=False)
        log.info(
            f"Backup file for usd/rub exchange rates was updated."
            f"New dates range: {rates.date.min()} : {rates.date.max()}"
        )

    return rates
