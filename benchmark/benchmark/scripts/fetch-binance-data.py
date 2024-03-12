import ccxt
import datetime
import pandas as pd
import pytz

exchange = ccxt.binance({
    'timeout': 30000  # Increase the timeout value to 30 seconds or more
})
pairs = ["BTC/USDT"]

timeframe = "5m"
timeframe_mins = {"5m": 5}

start_date = "2022-10-10"
end_date = "2023-11-10"


start_date_ts = pd.Timestamp(start_date, tz="UTC")
end_date_ts = pd.Timestamp(end_date, tz="UTC")

current_datetime = datetime.datetime.now().astimezone(pytz.utc)

end_date = min(pd.Timestamp(end_date, tz="UTC").to_pydatetime(), current_datetime.astimezone(pytz.utc))
print('End Date: ', end_date)
end_date_ts = pd.Timestamp(end_date)

for pair in pairs:
    start_time = int(start_date_ts.timestamp() * 1000)
    end_time = int(end_date_ts.timestamp() * 1000)
    flag = True
    data = pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])
    while flag:
        ohlcv = exchange.fetch_ohlcv(
            symbol=pair,
            timeframe=timeframe,
            since=start_time,
            limit=1000,
            params={"until": end_time},
        )
                
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        data = pd.concat([data, df])
        start_time = int(
            (data.datetime.iloc[-1] + pd.DateOffset(minutes=timeframe_mins[timeframe])).timestamp() * 1000
        )       
        
        flag = data.datetime.iloc[-1] <= end_date_ts - datetime.timedelta(minutes=15)
        print('Start window: ', data.datetime.iloc[-1])
        print("End window: ", end_date_ts - datetime.timedelta(minutes=15))

    data.to_csv(f"data/binance/{str(exchange).lower()}_{pair.replace('/', '-').lower()}_{timeframe}_candles_{start_date}.csv", index=False)
        