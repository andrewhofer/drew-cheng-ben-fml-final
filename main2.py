import numpy as np
import pandas as pd
import indicators as ind
from matplotlib import pyplot as pp
import scrape as scrap
import DeepQLearner as Q
import chatGPT as gpt

## model without chatGPT as indicator

num_train = 0
total_cum_list = []  # track total_cum at each training iteration
total_cum_oos_when_is_beat_bench = []  # track total_cum out of sample when in sample total_cum beats bench_cum

beat_bench_first_time_flag = False  # flag to check if total_cum beat bench_cum for the first time
train_data = pd.DataFrame(columns=['In_sample_total_cum', 'In_sample_holding_var', 'Out_sample_total_cum', 'Out_sample_holding_var'])


while True:  # added loop
    train_start = '2019-04-01'
    train_end = '2020-03-31'

    test_start = '2020-04-01'
    test_end = '2020-12-31'

    all_indicators = ['SMA_25', 'SMA_50', 'OBV', 'ADL', 'ADX', 'MACD', 'RSI', 'Sto_Osc']

    indicators = pd.read_csv('XLK_Inds.csv')
    indicators.set_index('Date', inplace=True)

    symbol = 'XLK'
    shares = 1000
    starting_cash = 200000

    # Define state and action dimensions
    state_dim = 8
    action_dim = 3

    # Initialize the DQN model and load indicators
    dqn = Q.DeepQLearner(state_dim=state_dim, action_dim=action_dim,alpha = 0.9, gamma = 0.5, epsilon = 0.998,
                      epsilon_decay = 0.9999, hidden_layers = 3, buffer_size = 150, batch_size = 64)

    prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
    prices['Trades'], prices['Holding'] = 0, 0
    fresh_frame = prices.copy()

    train_inds = indicators.loc[train_start:train_end]
    starting_stock_value = prices[symbol].iloc[0]

    days = 1
    flat_holding_penalty = 1
    num_trips=100

    # Training trips
    for i in range(num_trips):
        current_holding = 0
        data = fresh_frame.copy()
        cash = starting_cash
        prev_portfolio = starting_cash
        reward = 0
        stock_value_3_days_ago = starting_stock_value
        portfolio_3_days_ago = starting_cash
        prev_action = None

        # Loop over the data
        for j in range(len(train_inds)):
            state = []
            for indicator in all_indicators:
                state.append(train_inds[indicator].iloc[j])

            state = np.array(state)
            price = data[symbol].iloc[j]
            curr_portfolio = cash + (current_holding * price)


            weight = 0.2

            reward = ((curr_portfolio / prev_portfolio) / 1)*weight + (1-weight)*((curr_portfolio / starting_cash) - 1)

            if prev_action == 2:
                if reward > 0:
                    reward *= 2
                else:
                    reward *= 0.5

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
                #reward-=flat_holding_penalty
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

            prev_action = action
            prev_portfolio = curr_portfolio
            #print(f'Day {j}: Action {action}')

        # Get results of training trip
        #cum_frame, total_cum, adr, std = ind.assess_strategy(train_start, train_end, data, symbol, starting_cash)
        #print("Training trip " + str(i) + " net profit: $" + str(round(total_cum-starting_cash, 2)))

    cum_frame, total_cum, adr, std = ind.assess_strategy(train_start, train_end, data, symbol, starting_cash)
    total_cum_list.append(total_cum)  # record total_cum at each iteration

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
        #print(action)

        if action == 0:  # Buy
            if current_holding < shares:
                trade = shares
                current_holding += shares
                data.iloc[j, 1] = trade
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding
        elif action == 1:  # Sell
            if current_holding > -shares:
                trade = -shares
                current_holding -= shares
                data.iloc[j, 1] = trade
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding
        else:  # Flat
            if current_holding == shares: # Sell
                trade = shares
                current_holding = 0
                data.iloc[j, 1] = -shares
                data.iloc[j, 2] = current_holding
            elif current_holding == -shares: # Buy
                trade = shares
                current_holding = 0
                data.iloc[j, 1] = shares
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding

    # Get results of training trip
    cum_frame, total_cum, adr, std = ind.assess_strategy(train_start, train_end, data, symbol, starting_cash)

    # Get the variance of the holdings
    in_sample_holding_var = data['Holding'].var()


    prices = ind.get_data(train_start, train_end, [symbol], include_spy=False)
    prices['Trades'], prices['Holding'] = 0, shares
    prices['Trades'].iloc[0] = shares
    bench_frame, bench_cum, bench_adr, bench_std = ind.assess_strategy(train_start, train_end, prices, symbol, starting_cash)

    print("***IN SAMPLE TEST RESULTS***")
    print(total_cum, adr, std)
    print(bench_cum, bench_adr, bench_std)
    print("***IN SAMPLE TEST RESULTS***")

    pp.plot(bench_frame, color='r', label='Buy and Hold Benchmark')  # Benchmark
    pp.plot(cum_frame, color='b', label='In Sample Q–Learned Strategy')
    pp.legend()
    pp.title("Benchmark vs In Sample Q–Learned Strategy no GPT on run "+str(num_train))
    pp.xlabel("Date")
    pp.ylabel("Cumulative Returns")
    pp.grid()
    pp.show()
    num_train+=1

    if total_cum > bench_cum:
        if not beat_bench_first_time_flag:
            print(f'total_cum > bench_cum for the first time after {num_train} training iterations')
            beat_bench_first_time_flag = True

    ##############################################################################
    """OUT OF SAMPLE TESTING"""
    test_prices = ind.get_data(test_start, test_end, [symbol], include_spy=False)
    test_prices['Trades'], test_prices['Holding'] = 0, 0
    data = test_prices.copy()

    test_inds = indicators.loc[test_start:test_end]

    # Resetting cash and current_holding for out-of-sample testing
    cash = starting_cash
    current_holding = 0

    # Loop over the data
    for j in range(len(test_inds)):
        state = []
        for indicator in all_indicators:
            state.append(test_inds[indicator].iloc[j])

        state = np.array(state)
        price = data[symbol].iloc[j]  # Updating the price for the current day
        action = dqn.test(state)

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
            if current_holding == shares:  # Sell
                trade = shares
                trade_val = price * trade
                cash += trade_val
                current_holding = 0
                data.iloc[j, 1] = -shares
                data.iloc[j, 2] = current_holding
            elif current_holding == -shares:  # Buy
                trade = shares
                trade_val = price * trade
                cash -= trade_val
                current_holding = 0
                data.iloc[j, 1] = shares
                data.iloc[j, 2] = current_holding
            else:
                data.iloc[j, 1] = 0
                data.iloc[j, 2] = current_holding

    cum_frame, total_cum_out, adr, std = ind.assess_strategy(test_start, test_end, data, symbol, starting_cash)

    prices = ind.get_data(test_start, test_end, [symbol], include_spy=False)
    prices['Trades'], prices['Holding'] = 0, shares
    prices['Trades'].iloc[0] = shares
    bench_frame, bench_cum, bench_adr, bench_std = ind.assess_strategy(test_start, test_end, prices, symbol,
                                                                       starting_cash)

    print("***OUT OF SAMPLE TEST RESULTS***")
    print(total_cum_out, adr, std)
    print(bench_cum, bench_adr, bench_std)
    print("***OUT OF SAMPLE TEST RESULTS***")

    pp.plot(bench_frame, color='r', label='Buy and Hold Benchmark')  # Benchmark
    pp.plot(cum_frame, color='b', label='Out of Sample Q–Learned Strategy')
    pp.legend()
    pp.title("Benchmark vs Out of Sample Q–Learned Strategy no GPT on run "+str(num_train))
    pp.xlabel("Date")
    pp.ylabel("Cumulative Returns")
    pp.grid()
    pp.show()

    # Get the variance of the holdings
    out_sample_holding_var = data['Holding'].var()
    # ... Your code here...

    # Append the results to the DataFrame
    new_row = pd.DataFrame({'In_sample_total_cum': [total_cum],
                            'In_sample_holding_var': [in_sample_holding_var],
                            'Out_sample_total_cum': [total_cum_out],
                            'Out_sample_holding_var': [out_sample_holding_var]})

    train_data = pd.concat([train_data, new_row]).reset_index(drop=True)

    if num_train == 50:
        break


print(f'Number of training iterations: {num_train}')
print(f'Average total_cum after {num_train} training iterations: {np.mean(total_cum_list)}')

if total_cum_oos_when_is_beat_bench:  # check if the list is not empty
    print(f'Average total_cum for out-of-sample testing when in-sample total_cum beats bench_cum: {np.mean(total_cum_oos_when_is_beat_bench)}')



print(train_data)

train_data.to_csv('train_data_no.csv', index=False)

