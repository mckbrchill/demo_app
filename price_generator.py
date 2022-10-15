import pandas as pd
import numpy as np
import time
import os
from random import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PriceGenerator:

    def __init__(self, tick_size = 1):
        self.tick_size = tick_size


    def generate_movement(self):
        movement = -1 if random() < 0.5 else 1
        return movement


    def apply_movement(self, val):
        price_change = self.generate_movement()
        val += price_change

        return val


    def generate_ohlc(self, tickers, start_prices, startDate, seed=None):
        np.random.seed(seed)

        P = list(np.around(start_price + np.random.normal(scale=4.0, size=4).cumsum(), 4) for start_price in start_prices)
        P = np.asarray(P).flatten()
        df = pd.DataFrame({
            'ticker': sorted(tickers[0] * 4),
            'price': P,
        },
            index=list(pd.date_range(startDate, periods=4, freq='L')) * 100
        )

        df = df.groupby('ticker')['price'].ohlc()

        df['price'] = start_prices
        iterables = [[startDate], df.index]

        index = pd.MultiIndex.from_product(iterables, names=["datetime", "ticker"])
        df = pd.DataFrame(index=index, data=df.values, columns=df.columns)

        return df

    def load_data(self, skiprows = 0):
        dfs = {}
        if os.path.isfile("./tickers_data.csv"):
            # print(skiprows)
            df = pd.read_csv('tickers_data.csv', sep=';', skiprows = range(1,skiprows+1))
            tickers = df['ticker'].unique()

            for ticker in tickers:
                tdf = df.loc[df['ticker'] == ticker].copy()
                tdf.set_index(pd.to_datetime(tdf['datetime']), inplace=True)
                del tdf['datetime']
                del tdf['ticker']
                dfs[ticker] = tdf.copy()
        else:
            tickers = ['ticker{}'.format(i) for i in ["%.2d" % i for i in range(100)]]
            dfs = {ticker: pd.DataFrame(columns = ['open', 'high', 'low', 'close', 'price'],
                                       index = pd.DatetimeIndex([], name = 'datetime')) for ticker in tickers}

        return dfs

    def run(self):

        if os.path.isfile("./tickers_data.csv"):
            df = pd.read_csv('tickers_data.csv', sep=';', index_col=[0, 1])
            df = df.tail(100)
            tickers = [list(df.index.levels[1].values)]

        else:
            startDate = pd.to_datetime(time.time() * 1e9).replace(microsecond=0)
            tickers = [['ticker{}'.format(i) for i in ["%.2d" % i for i in range(100)]]]
            start_prices = 100 + np.random.randn(1, 100)[0]

            df = self.generate_ohlc(tickers, start_prices, startDate)
            df.to_csv('tickers_data.csv', sep=';')
            time.sleep(1)

        while True:
            try:
                df_n = None

                if len(df.index.levels[0]) > 60 * 60:
                    df.drop(df.index.levels[0][0], level='datetime', inplace=True)

                prev_prices = df.tail(100)['price'].values

                new_time = pd.to_datetime(time.time() * 1e9).replace(microsecond=0)
                new_prices = df.tail(100)['price'].apply(lambda x: self.apply_movement(x)).copy()
                new_prices = new_prices.values

                df_n = self.generate_ohlc(tickers, new_prices, new_time)

                with open('tickers_data.csv', 'a', newline='') as f:
                    df_n.to_csv(f, header=f.tell() == 0, sep=';')

            except KeyboardInterrupt:
                if df_n is not None:
                    with open('tickers_data.csv', 'a', newline='\n') as f:
                        df_n.to_csv(f, header=f.tell() == 0, sep=';')

            df = df_n.copy()
            time.sleep(self.tick_size)

