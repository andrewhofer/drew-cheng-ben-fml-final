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

indicators = pd.read_csv('XLK_Inds.csv')
indicators.set_index('Date', inplace=True)

symbol = 'XLK'
shares = 1000
starting_cash = 200000

# Define state and action dimensions
state_dim = 9
action_dim = 3

# Initialize the DQN model and load indicators
dqn = Q.DeepQLearner(state_dim=state_dim, action_dim=action_dim,alpha = 0.9, gamma = 0.9, epsilon = 0.998,
                  epsilon_decay = 0.999, hidden_layers = 4, buffer_size = 150, batch_size = 64)

prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
prices['Trades'], prices['Holding'] = 0, 0
fresh_frame = prices.copy()

train_inds = indicators.loc[train_start:train_end]
starting_stock_value = prices[symbol].iloc[0]

days = 10
flat_holding_penalty = 10

# Training trips
for i in range(5):
    current_holding = 0
    data = fresh_frame.copy()
    cash = starting_cash
    prev_portfolio = starting_cash
    reward = 0
    stock_value_3_days_ago = starting_stock_value
    portfolio_3_days_ago = starting_cash

    # Loop over the data
    for j in range(len(train_inds)):
        state = []
        for indicator in all_indicators:
            state.append(train_inds[indicator].iloc[j])

        state = np.array(state)
        price = data[symbol].iloc[j]
        curr_portfolio = cash + (current_holding * price)
        if j >= days:
            stock_value_3_days_ago = data[symbol].iloc[j - days]
            portfolio_3_days_ago = cash + (data['Holding'].iloc[j - days] * data[symbol].iloc[j - days])

        #reward = ((curr_portfolio / starting_cash) - 1)*100
        #reward = ((curr_portfolio / starting_cash) - (price / starting_stock_value)) * 20
        #reward = -((curr_portfolio / starting_cash) - (price / starting_stock_value)) ** 2 * 20
        #performance_diff = (curr_portfolio / starting_cash) - (price / starting_stock_value)
        performance_diff = (curr_portfolio / portfolio_3_days_ago) - (price / stock_value_3_days_ago)
        # ...

        if performance_diff < 0:  # Portfolio is underperforming
            reward = performance_diff ** 2 * -60
        else:  # Portfolio is outperforming or equal
            reward = performance_diff ** 2 * 200
        reward += ((curr_portfolio / starting_cash) - 1)*3
        #reward *= j/10
        print(f'reward {reward}')

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
            reward-=flat_holding_penalty
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

    # Get results of training trip
    cum_frame, total_cum, adr, std = ind.assess_strategy(train_start, train_end, data, symbol, starting_cash)
    print("Training trip " + str(i) + " net profit: $" + str(round(total_cum-starting_cash, 2)))

##############################################################################
"""IN SAMPLE TESTING"""
test_prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
test_prices['Trades'], test_prices['Holding'] = 0, 0
data = test_prices.copy()
current_holding = 0

test_inds = indicators.loc[train_start:train_end]

# Loop over the data
for j in range(len(test_inds)):
    state = []
    for indicator in all_indicators:
        state.append(test_inds[indicator].iloc[j])

    state = np.array(state)
    action = dqn.test(state)
    print(action)

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

print(data)

# Get results of training trip
cum_frame, total_cum, adr, std = ind.assess_strategy(train_start, train_end, data, symbol, starting_cash)

prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
prices = (prices / prices[symbol].iloc[0]) - 1  # Benchmark
pp.plot(prices, color='r', label='Buy and Hold Benchmark')  # Benchmark
pp.plot(cum_frame, color='b', label='In Sample Q–Learned Strategy')
pp.legend()
pp.title("Benchmark vs In Sample Q–Learned Strategy")
pp.xlabel("Date")
pp.ylabel("Cumulative Returns")
pp.grid()
pp.show()
##############################################################################
"""OUT OF SAMPLE TESTING"""
test_prices = ind.get_data(test_start, test_end, [symbol], include_spy=False)
test_prices['Trades'], test_prices['Holding'] = 0, 0
data = test_prices.copy()

test_inds = indicators.loc[test_start:test_end]
current_holding = 0

# Loop over the data
for j in range(len(test_inds)):
    state = []
    for indicator in all_indicators:
        state.append(test_inds[indicator].iloc[j])

    state = np.array(state)
    action = dqn.test(state)
    print(action)

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

print(data)
cum_frame, total_cum, adr, std = ind.assess_strategy(test_start, test_end, data, symbol, starting_cash)

prices = ind.get_data(test_start, test_end, [symbol], include_spy=False)
prices = (prices / prices[symbol].iloc[0]) - 1  # Benchmark
pp.plot(prices, color='r', label='Buy and Hold Benchmark')  # Benchmark
pp.plot(cum_frame, color='b', label='Out of Sample Q–Learned Strategy')
pp.legend()
pp.title("Benchmark vs Out of Sample Q–Learned Strategy")
pp.xlabel("Date")
pp.ylabel("Cumulative Returns")
pp.grid()
pp.show()