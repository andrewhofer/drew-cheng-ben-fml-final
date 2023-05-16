# yfinance requires pandas 1.3.5, breaks with 1.4.0
import yfinance as yf

tickerStrings = ['COMT','XLK','XLF']

for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period="max", interval="1d")
    data.to_csv(f'final_data/{ticker}.csv', float_format='%.2f')