import numpy as np
import indicators as ind
import matplotlib as pp
import scrape as scrap
import DeepQLearner as Q

# Initialize the indicator class
indicators = ind.TechnicalIndicators()

start_date = '2019-01-01'
end_date = '2020-12-31'
symbol = 'XLK'
shares = 1000

df = ind.get_data(start_date, end_date, [symbol], include_spy=False)
df['High'] = ind.get_data(start_date, end_date, [symbol], column_name='High', include_spy=False)
df['Low'] = ind.get_data(start_date, end_date, [symbol], column_name='Low', include_spy=False)
df['Close'] = ind.get_data(start_date, end_date, [symbol], column_name='Close', include_spy=False)
df['Volume'] = ind.get_data(start_date, end_date, [symbol], column_name='Volume', include_spy=False)
indicators.data = df
indicators.symbol = symbol

# Add indicators
indicators.add_sma(25)
indicators.add_sma(50)
indicators.add_obv()
indicators.add_adl()
indicators.add_adx(14)
indicators.add_macd(12, 26, 9)
indicators.add_rsi(14)
indicators.add_stochastic_oscillator(14)
indicators.data = indicators.data.dropna()
indicators.normalize()
del indicators.data['High']
del indicators.data['Low']
del indicators.data['Close']
del indicators.data['Volume']
del indicators.data[symbol]

for j in range(len(indicators.data)):
    curr_day = indicators.data.iloc[[j]]
    year = str(curr_day.index.year.tolist()[0])
    month = str(curr_day.index.month.tolist()[0])
    day = str(curr_day.index.day.tolist()[0])
    lines = scrap.gather_headlines(year, month, day)
    print(lines)



indicators.data.to_csv('test.csv', index=True)

# Define state and action dimensions
state_dim = 8
action_dim = 3
# Initialize the DQN model
dqn = Q.DeepQLearner(state_dim=state_dim, action_dim=action_dim)

prices = ind.get_data(start_date, end_date, [symbol], include_spy=False)
prices['Trades'], prices['Holding'] = 0, 0
fresh_frame = prices.copy()
for i in range(500):
    current_holding = 0
    data = fresh_frame.copy()
    cash = 200000
    prev_portfolio = 200000
    reward = 0

    # Loop over the data
    for j in range(len(indicators.data)):
        state = []
        for indicator in ['SMA_25', 'SMA_50', 'OBV', 'ADL', 'ADX', 'MACD', 'RSI', 'Sto_Osc']:
            state.append(indicators.get_indicator(indicator, j))

        state = np.array(state)
        price = data[symbol].iloc[j]
        position_value = current_holding * price
        reward = position_value + cash - 200000

        """
        if j == 0:
            action = dqn.test(state)
        else:
        """
        action = dqn.train(state, reward)

        if action == 0:  # Buy
            if current_holding < shares:
                trade = shares
                trade_val = price * trade
                cash -= trade_val
                current_holding += shares
                data.iloc[j, 1] = trade
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding
        elif action == 1:  # Sell
            if current_holding > -shares:
                trade = -shares
                trade_val = price * abs(trade)
                cash += trade_val
                current_holding -= shares
                data.iloc[j, 1] = trade
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding
        else:  # Flat
            if current_holding == shares: # Sell
                trade = shares
                trade_val = price * trade
                cash += trade_val
                current_holding = 0
                data.iloc[j, 1] = -shares
                data.iloc[j, 2] = current_holding
            elif current_holding == -shares: # Buy
                trade = shares
                trade_val = price * trade
                cash -= trade_val
                current_holding = 0
                data.iloc[j, 1] = shares
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding

        # Print the action (just for debugging)
        print(f'Day {j}: Action {action}')
    cum_frame, total_cum, adr, std = ind.assess_strategy(start_date, end_date, data, symbol, 200000)
    print("Training trip " + str(j) + " net profit: $" + str(round(total_cum-200000, 2)))