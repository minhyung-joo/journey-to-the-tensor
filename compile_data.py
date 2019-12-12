import pandas as pd
import requests
import time
from io import StringIO

def trim_date(date_str):
    return date_str[:10]

price_data = pd.read_csv('prices.csv')
price_data["date"] = price_data["date"].map(trim_date)
print (price_data.head())
symbols = price_data["symbol"].drop_duplicates()
print (len(symbols))
headers = {
    "cookie": "A1=d=AQABBLm18V0CEIgiyygnUNsCtEY9QPOGjEIFEgEBAQEH8137XQAAAAAA_SMAAAcItrXxXYs9EDk&S=AQAAAhn3bLFwxOxSqjHvWkRLRBQ; A3=d=AQABBLm18V0CEIgiyygnUNsCtEY9QPOGjEIFEgEBAQEH8137XQAAAAAA_SMAAAcItrXxXYs9EDk&S=AQAAAhn3bLFwxOxSqjHvWkRLRBQ; A1S=d=AQABBLm18V0CEIgiyygnUNsCtEY9QPOGjEIFEgEBAQEH8137XQAAAAAA_SMAAAcItrXxXYs9EDk&S=AQAAAhn3bLFwxOxSqjHvWkRLRBQ&j=WORLD; GUC=AQEBAQFd8wdd-0IdnwSF; PRF=t%3DTSLA; APID=UP92e6f7d8-1c90-11ea-aa22-022b7e107f5e; APIDTS=1576121816; B=3i41thdev3ddm&b=3&s=fh"
}
for symbol in symbols:
    link = "https://query1.finance.yahoo.com/v7/finance/download/" + symbol + "?period1=1481514185&period2=1576121787&interval=1d&events=history&crumb=76tc50nKkov"
    r = requests.get(url = link, headers=headers)
    response_txt = StringIO(r.text.lower())
    recent_data = pd.read_csv(response_txt)
    recent_data.insert(1, "symbol", symbol)
    if 'adj close' in recent_data.columns:
        recent_data = recent_data.drop(columns=['adj close'])
    
    recent_data = recent_data.reindex(columns=['date', 'symbol', 'open', 'close', 'low', 'high', 'volume'])
    price_data = pd.concat([price_data, recent_data])
    time.sleep(0.5)

price_data.to_csv("new_prices.csv", index=False)