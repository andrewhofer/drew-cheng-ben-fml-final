import pandas as pd
import numpy as np
import indicators as ind
import DeepQLearner as Q

# Initialize the indicator class
indicators = ind.TechnicalIndicators()
df = ind.get_data('2010-01-01', '2010-12-31', ['XLK'], include_spy=False)
df['High'] = ind.get_data('2010-01-01', '2010-12-31', ['XLK'], column_name='High', include_spy=False)
df['Low'] = ind.get_data('2010-01-01', '2010-12-31', ['XLK'], column_name='Low', include_spy=False)
df['Close'] = ind.get_data('2010-01-01', '2010-12-31', ['XLK'], column_name='Close', include_spy=False)
df['Volume'] = ind.get_data('2010-01-01', '2010-12-31', ['XLK'], column_name='Volume', include_spy=False)
indicators.data = df
indicators.symbol = 'XLK'

# Add indicators
indicators.add_sma(50)
indicators.add_sma(200)
indicators.add_obv()
indicators.add_adl()
indicators.add_adx(14)
indicators.add_macd(12, 26, 9)
indicators.add_rsi(14)
indicators.add_stochastic_oscillator(14)
print(indicators.data)

# Define state and action dimensions
state_dim = 9
action_dim = 3

# Initialize the DQN model
dqn = Q.DeepQLearner(state_dim=state_dim, action_dim=action_dim)


# # Loop over the data
# for i in range(state_dim, len(data)):
#     # Prepare the state
#     state = []
#     for indicator in ['SMA_50', 'SMA_200', 'OBV', 'ADL', 'ADX', 'MACD', 'RSI', 'Stochastic_Oscillator']:
#         state.append(indicators.get_indicator(indicator, i))
#
#     state = np.array(state)
#
#     # Get the reward (you need to define your own reward function)
#     #reward = P.calculate_reward(i, data)  # Implement this function
#
#     # Train the DQN model
#     #action = dqn.train(state, reward)
#
#     # Print the action (just for debugging)
#     #print(f'Day {i}: Action {action}')
