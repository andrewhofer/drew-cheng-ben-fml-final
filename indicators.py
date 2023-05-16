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
        return self.data[indicator_name].iloc[day]

    def normalize(self):
        norm_25 = (self.data['SMA_25'] - self.data['SMA_25'].mean()) / self.data['SMA_25'].std()
        self.data['SMA_25'] = norm_25
        norm_50 = (self.data['SMA_50'] - self.data['SMA_50'].mean()) / self.data['SMA_50'].std()
        self.data['SMA_50'] = norm_50
        norm_obv = (self.data['OBV'] - self.data['OBV'].mean()) / self.data['OBV'].std()
        self.data['OBV'] = norm_obv
        norm_adl = (self.data['ADL'] - self.data['ADL'].mean()) / self.data['ADL'].std()
        self.data['ADL'] = norm_adl
        norm_adx = (self.data['ADX'] - self.data['ADX'].mean()) / self.data['ADX'].std()
        self.data['ADX'] = norm_adx
        norm_mac = (self.data['MACD'] - self.data['MACD'].mean()) / self.data['MACD'].std()
        self.data['MACD'] = norm_mac
        norm_rsi = (self.data['RSI'] - self.data['RSI'].mean()) / self.data['RSI'].std()
        self.data['RSI'] = norm_rsi
        norm_osc = (self.data['Sto_Osc'] - self.data['Sto_Osc'].mean()) / self.data['Sto_Osc'].std()
        self.data['Sto_Osc'] = norm_osc


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

def assess_strategy(start, end, trades, symbol, starting_value):

    num_trades = 0
    trades['CASH'] = 0
    trades['PORTFOLIO'] = 0
    prev_cash = starting_value

    print(trades)

    for i in range(len(trades)):
        curr_trade = trades.iloc[i, 1]
        curr_price = trades.iloc[i, 0]

        if curr_trade != 0:
            num_trades += 1
            if curr_trade > 0:
                capital = curr_trade * curr_price
                #cost = (capital + self.fixed_cost + capital*self.floating_cost)
                trades.iloc[i, 3] = prev_cash - capital
                prev_cash = prev_cash - capital
            if curr_trade < 0:
                capital = -curr_trade * curr_price
                #cost = (capital - self.fixed_cost - capital*self.floating_cost)
                trades.iloc[i:, 3] = prev_cash + capital
                prev_cash = prev_cash + capital
        else:
            trades.iloc[i, 3] = prev_cash
        trades.iloc[i, 4] = trades.iloc[i, 3] + (trades.iloc[i, 0] * trades.iloc[i, 2])

    cum_frame = (trades['PORTFOLIO'] / 200000) - 1
    adr = ((trades['PORTFOLIO'] / trades['PORTFOLIO'].shift()) - 1).mean()
    std_dr = ((trades['PORTFOLIO'] / trades['PORTFOLIO'].shift()) - 1).std()
    total_cum = trades.iloc[-1, -1]

    return cum_frame, total_cum, adr, std_dr