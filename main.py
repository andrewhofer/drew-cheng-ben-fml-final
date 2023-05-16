import numpy as np
import pandas as pd
import indicators as ind
from matplotlib import pyplot as pp
import scrape as scrap
import DeepQLearner as Q
import chatGPT as gpt

train_start = '2019-04-01'
train_end = '2020-06-30'

test_start = '2020-07-01'
test_end = '2020-12-31'

all_indicators = ['SMA_25', 'SMA_50', 'OBV', 'ADL', 'ADX', 'MACD', 'RSI', 'Sto_Osc', 'GPT Sent']

symbol = 'XLK'
shares = 1000
starting_cash = 200000

# Define state and action dimensions
state_dim = 9
action_dim = 3

# Initialize the DQN model and load indicators
dqn = Q.DeepQLearner(state_dim=state_dim, action_dim=action_dim)
indicators = pd.read_csv('XLK_Inds.csv')

prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
prices['Trades'], prices['Holding'] = 0, 0
fresh_frame = prices.copy()
indicators.set_index('Date', inplace=True)

train_inds = indicators.loc[train_start:train_end]

# Training trips
for i in range(1):
    current_holding = 0
    data = fresh_frame.copy()
    cash = starting_cash
    prev_portfolio = starting_cash
    reward = 0

    # Loop over the data
    for j in range(len(train_inds)):
        state = []
        for indicator in all_indicators:
            state.append(train_inds[indicator].iloc[j])

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

        #print(f'Day {j}: Action {action}')

    # Get results of training trip
    cum_frame, total_cum, adr, std = ind.assess_strategy(train_start, train_end, data, symbol, starting_cash)
    print("Training trip " + str(i) + " net profit: $" + str(round(total_cum-starting_cash, 2)))

prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
prices = (prices / prices[symbol].iloc[0]) - 1 # Benchmark
pp.plot(prices, color='m', label='Buy and Hold Benchmarl') # Benchmark
pp.plot(cum_frame, color='g', label='Q–Learned Strategy')
pp.legend()
pp.title("Final test run vs. Benchmark")
pp.xlabel("Date")
pp.ylabel("Cumulative Returns")
pp.grid()
pp.show()