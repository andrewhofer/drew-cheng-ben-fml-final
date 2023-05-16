import numpy as np
import pandas as pd
import indicators as ind
from matplotlib import pyplot as pp
import scrape as scrap
import DeepQLearner as Q
import chatGPT as gpt

# Initialize the indicator class
# indicators = ind.TechnicalIndicators()

start_date = '2019-03-14'
end_date = '2020-12-31'
symbol = 'XLK'
shares = 1000
starting_cash = 200000

# Define state and action dimensions
state_dim = 9
action_dim = 3
# Initialize the DQN model
dqn = Q.DeepQLearner(state_dim=state_dim, action_dim=action_dim)
indicators = pd.read_csv('test.csv')
sent = (indicators['GPT Sent'] - indicators['GPT Sent'].mean()) / indicators['GPT Sent'].std()
indicators['GPT Sent'] = sent

prices = ind.get_data(start_date, end_date, [symbol], include_spy=False)
prices['Trades'], prices['Holding'] = 0, 0
fresh_frame = prices.copy()
# Training trips
cum_frame = 0
for i in range(50):
    current_holding = 0
    data = fresh_frame.copy()
    cash = starting_cash
    prev_portfolio = starting_cash
    reward = 0

    # Loop over the data
    for j in range(len(indicators)):
        state = []
        for indicator in ['SMA_25', 'SMA_50', 'OBV', 'ADL', 'ADX', 'MACD', 'RSI', 'Sto_Osc', 'GPT Sent']:
            state.append(indicators[indicator].iloc[j])

        state = np.array(state)
        price = data[symbol].iloc[j]
        curr_portfolio = cash + (current_holding * price)
        reward = (curr_portfolio / starting_cash) - 1

        if j == 0:
            action = dqn.test(state)
        else:
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

        print(f'Day {j}: Action {action}')
    cum_frame, total_cum, adr, std = ind.assess_strategy(start_date, end_date, data, symbol, starting_cash)
    print("Training trip " + str(i) + " net profit: $" + str(round(total_cum-starting_cash, 2)))

prices = ind.get_data(start_date, end_date, [symbol], include_spy=False)
prices = (prices / prices[symbol].iloc[0]) - 1
pp.plot(prices, color='b', label='XLK')
pp.plot(cum_frame, color='r', label='Q–Learned Strategy')
pp.legend()
pp.title("Final test run vs. Benchmark")
pp.xlabel("Date")
pp.ylabel("Cumulative Returns")
pp.grid()
pp.show()