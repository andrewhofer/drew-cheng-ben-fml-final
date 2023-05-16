import numpy as np
import indicators as ind
import matplotlib as pp
import scrape as scrap
import DeepQLearner as Q
import chatGPT as gpt

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

indicators.data['GPT Sent'] = 0

for j in range(len(indicators.data)):
    curr_day = indicators.data.iloc[[j]]
    year = str(curr_day.index.year.tolist()[0])
    month = str(curr_day.index.month.tolist()[0])
    day = str(curr_day.index.day.tolist()[0])
    lines = scrap.gather_headlines(year, month, day)
    score = 0
    if lines is not None:
        score = gpt.process_titles_average_score(lines)

    indicators.data['GPT Sent'].iloc[j] = score

    print(str(j) + " of " + str(len(indicators.data)) + " done.")

indicators.data.to_csv('test.csv', index=True)