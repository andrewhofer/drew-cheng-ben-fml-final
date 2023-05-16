import pandas as pd
import ta

class TechnicalIndicators:

    def __init__(self):
        self.data = None
        self.symbol = None

    def add_sma(self, window_size: int):
        """Add SMA to the data."""
        self.data[f'SMA_{window_size}'] = ta.trend.sma_indicator(self.data[self.symbol], window_size)

    def add_obv(self):
        """Add On-balance Volume to the data."""
        self.data['OBV'] = ta.volume.on_balance_volume(self.data[self.symbol], self.data['Volume'])

    def add_adl(self):
        """Add Accumulation/Distribution Line to the data."""
        self.data['ADL'] = ta.volume.acc_dist_index(self.data['High'], self.data['Low'], self.data[self.symbol], self.data['Volume'])

    def add_adx(self, window_size: int):
        """Add Average Directional Index to the data."""
        self.data['ADX'] = ta.trend.adx(self.data['High'], self.data['Low'], self.data[self.symbol], window_size)

    def add_macd(self, window_size_short: int, window_size_long: int, window_size_signal: int):
        """Add MACD to the data."""
        self.data['MACD'] = ta.trend.macd_diff(self.data[self.symbol], window_size_short, window_size_long, window_size_signal)

    def add_rsi(self, window_size: int):
        """Add Relative Strength Index to the data."""
        self.data['RSI'] = ta.momentum.rsi(self.data[self.symbol], window_size)

    def add_stochastic_oscillator(self, window_size: int):
        """Add Stochastic Oscillator to the data."""
        self.data['Sto_Osc'] = ta.momentum.stoch(self.data['High'], self.data['Low'], self.data[self.symbol], window_size)

    def get_indicator(self, indicator_name: str, day: int):
        """Get the value of a given indicator at a given day."""
        return self.data.loc[day, indicator_name]

def get_data(start_date, end_date, symbols, column_name='Adj Close', include_spy=True):
    """
    Gets adjusted close data from start_date to end_date for the requested 
    symbols and SPY. Returns a pandas dataframe with the data.
    """
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    # Read SPY.
    df_spy = pd.read_csv(f'data/SPY.csv', index_col=['Date'], parse_dates=True, \
        na_values=['nan'], usecols=['Date',column_name])

    # Use SPY to eliminate non-market days.
    df = df.join(df_spy, how='inner')
    df = df.rename(columns={column_name:'SPY'})

    # Append the data for the remaining symbols, retaining all market-open days.
    for sym in symbols:
        df_sym = pd.read_csv(f'final_data/{sym}.csv', index_col=['Date'], parse_dates=True, \
            na_values=['nan'], usecols=['Date',column_name])
        df = df.join(df_sym, how='left')
        df = df.rename(columns={column_name:sym})

    # Eliminate SPY if requested.
    if not include_spy: 
        del df['SPY']

    return df