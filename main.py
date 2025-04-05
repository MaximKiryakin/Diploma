

from utils.logger import Logger

log = Logger(__name__).get_logger()

from utils.portfolio import Portfolio


calc = Portfolio(
    dt_calc='30.10.2024',
    dt_start='03.11.2019',
    stocks_step=10,
    tickers_list= [
        'GAZP', 'LKOH', 'ROSN',
        'SBER', 'VTBR',
        'GMKN', 'NLMK', 'RUAL',
        'MTSS', 'RTKM',
        'MGNT',  'LNTA',
    ]
)
calc= calc.load_stock_data(use_backup_data=False).load_multipliers().create_portfolio()

