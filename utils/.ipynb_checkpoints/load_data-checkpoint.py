def download_finam_quotes(
    ticker: str,
    period: int,
    start: str,
    end: str,
    filename: str = "quotes.txt",
    bucket_size: int = 100
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
        filename: Output filename (optional)
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
        2: timedelta(minutes=1), 3: timedelta(minutes=5), 4: timedelta(minutes=10),
        5: timedelta(minutes=15), 6: timedelta(minutes=30), 7: timedelta(hours=1),
        8: timedelta(days=1), 9: timedelta(weeks=1), 10: timedelta(days=30),
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
    result_df = None
    first_bucket = True

    while current_start <= end_date:
        start_rev = current_start.strftime("%Y%m%d")
        end_rev = min(current_start + delta * bucket_size, end_date).strftime("%Y%m%d")

        params = urlencode([
            ('market', 0),    # Market type (0 - stocks)
            ('em', tickers[ticker]),
            ('code', ticker),
            ('apply', 0),
            ('df', current_start.day),
            ('mf', current_start.month - 1),
            ('yf', current_start.year),
            ('from', current_start.date()),
            ('dt', end_date.day),
            ('mt', end_date.month - 1),
            ('yt', end_date.year),
            ('to', end_date.date()),
            ('p', period),
            ('f', f"{ticker}_{start_rev}_{end_rev}"),
            ('e', ".csv"),
            ('cn', ticker),
            ('dtf', 1),       # Date format (1 - YYYYMMDD)
            ('tmf', 1),       # Time format (1 - HHMMSS)
            ('MSOR', 0),      # Candle time (0 - open; 1 - close)	
            ('mstime', "on"), # Moscow time zone
            ('mstimever', 1),
            ('sep', 1),
            ('sep2', 1),
            ('datf', 1),
            ('at', 1 if first_bucket else 0)
        ])

        url = f"http://export.finam.ru/{ticker}_{start_rev}_{end_rev}.csv?{params}"

        try:
            with urlopen(url) as response:
                content = response.read().decode('utf-8')
                lines = [line.split(',') for line in content.splitlines()]
                
                if not lines:
                    break

                if first_bucket:
                    result_df = pd.DataFrame(lines[1:], columns=lines[0])
                    first_bucket = False
                else:
                    result_df = pd.concat([result_df, pd.DataFrame(lines[1:])])

                last_date = datetime.strptime(
                    f"{lines[-1][2]} {lines[-1][3]}",
                    "%Y%m%d %H%M%S"
                )
                current_start = last_date + delta

        except Exception as e:
            warnings.warn(f"Download error: {str(e)}")
            break

    return result_df