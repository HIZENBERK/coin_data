import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# RSI 계산 함수
def calculate_rsi(data, window=14):
    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD 계산 함수
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# 볼린저 밴드 계산 함수
def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

# 변동성 계산 함수
def calculate_volatility(data, window=14):
    return data['close'].pct_change().rolling(window=window).std()

# 가격 변동률 계산 함수
def calculate_price_change(data):
    return data['close'].pct_change()

# Upbit API를 사용하여 캔들 데이터를 가져오는 함수
def fetch_upbit_candles(market, interval, start_time, end_time):
    base_url = f"https://api.upbit.com/v1/candles/minutes/{interval}"
    headers = {"Accept": "application/json"}
    results = []

    while start_time < end_time:
        query_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
        params = {"market": market, "to": query_time, "count": 200}

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 429:
            time.sleep(1)
            continue
        elif response.status_code != 200:
            print(f"Error fetching data for {market}: {response.status_code}")
            break

        data = response.json()
        if not data:
            break

        for item in data:
            results.append({
                "timestamp": item["candle_date_time_kst"],
                "open": item["opening_price"],
                "high": item["high_price"],
                "low": item["low_price"],
                "close": item["trade_price"],
                "volume": item["candle_acc_trade_volume"],
                "value": item["candle_acc_trade_price"],
                "ticker": market
            })

        end_time = datetime.strptime(data[-1]["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S") - timedelta(seconds=1)

    return pd.DataFrame(results)

# 데이터 저장 경로 설정
base_path = r"F:\work space\coin\price_data"
os.makedirs(base_path, exist_ok=True)

# 티커와 설정
tickers = ["KRW-BTC", "KRW-ETH"]
interval = 60  # 60분 캔들
extra_hours = 26  # 기술 지표 계산에 필요한 추가 데이터
end_time = datetime.now()
start_time = end_time - timedelta(hours=60 + extra_hours)  # 추가 데이터 확보를 위해 시간 범위 확장

# 데이터 수집, 기술 지표 계산 및 저장
for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    data = fetch_upbit_candles(ticker, interval, start_time, end_time)
    
    if not data.empty:
        # 시간 순서 정렬 (오름차순)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(by='timestamp')

        # 기술 지표 계산
        data['RSI'] = calculate_rsi(data)
        data['MACD'], data['Signal'] = calculate_macd(data)
        data['MiddleBand'], data['UpperBand'], data['LowerBand'] = calculate_bollinger_bands(data)
        data['Volatility'] = calculate_volatility(data)
        data['PriceChange'] = calculate_price_change(data)

        # 최신 60시간 데이터만 남김
        data = data.iloc[-60:]

        # 데이터 저장
        output_file = os.path.join(base_path, f"{ticker}_with_tech_indicators.csv")
        data.to_csv(output_file, index=False)
        print(f"{ticker} 데이터 저장 완료: {output_file}")
    else:
        print(f"No data fetched for {ticker}.")

print("모든 데이터 수집 및 처리 완료.")
