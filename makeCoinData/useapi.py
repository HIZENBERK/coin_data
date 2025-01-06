import requests
import pandas as pd
from datetime import datetime, timedelta
import os


def fetch_upbit_candles(market: str, interval: str, start_date: str, end_date: str):
    """
    Upbit 캔들 데이터를 수집하여 Pandas DataFrame으로 반환.

    Args:
        market (str): 시장 (예: 'KRW-BTC').
        interval (str): 캔들 종류 ('minute1', 'minute10', 'minute60', 'day', 'week', 'month').
        start_date (str): 수집 시작 날짜 (YYYY-MM-DD).
        end_date (str): 수집 종료 날짜 (YYYY-MM-DD).

    Returns:
        pd.DataFrame: 캔들 데이터.
    """
    base_url = f"https://api.upbit.com/v1/candles/{interval}"
    headers = {"Accept": "application/json"}
    results = []

    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_time = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    while start_time < end_time:
        query_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "market": market,
            "to": query_time,
            "count": 200  # Upbit API는 한 번에 최대 200개까지 반환
        }

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error fetching data for {market} at {query_time}: {response.status_code}")
            break

        data = response.json()
        if len(data) == 0:
            break

        for item in data:
            results.append({
                "timestamp": item["candle_date_time_kst"],
                "open": item["opening_price"],
                "high": item["high_price"],
                "low": item["low_price"],
                "close": item["trade_price"],
                "volume": item["candle_acc_trade_volume"],
                "value": item["candle_acc_trade_price"]
            })

        # 마지막 데이터의 시간으로 이동
        start_time += timedelta(hours=len(data))

    return pd.DataFrame(results)


# KRW 거래 가능한 모든 코인 티커 가져오기
response = requests.get("https://api.upbit.com/v1/market/all", headers={"Accept": "application/json"})
markets = response.json()
krw_tickers = [market["market"] for market in markets if market["market"].startswith("KRW-")]
print(f"총 {len(krw_tickers)}개의 코인을 찾았습니다.")

# 데이터 저장 폴더 생성
output_dir = "../price_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 연도별 데이터 수집
start_year = 2019
end_year = datetime.now().year

for year in range(start_year, end_year + 1):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    yearly_data = []

    for ticker in krw_tickers:
        print(f"Fetching data for {ticker} in {year}...")
        df = fetch_upbit_candles(ticker, "minute60", start_date, end_date)

        if not df.empty:
            df["ticker"] = ticker
            yearly_data.append(df)

    if yearly_data:
        combined_df = pd.concat(yearly_data)
        combined_df.to_csv(f"{output_dir}/krw_coin_prices_{year}.csv", index=False)
        print(f"{year}년 데이터 저장 완료.")
    else:
        print(f"{year}년 데이터가 없습니다.")

print("모든 데이터 수집 및 저장이 완료되었습니다.")
