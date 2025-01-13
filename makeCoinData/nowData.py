import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import time

def fetch_upbit_candles(market: str, interval: int, start_time: datetime, end_time: datetime):
    """
    Upbit 캔들 데이터를 수집하여 Pandas DataFrame으로 반환.

    Args:
        market (str): 시장 (예: 'KRW-BTC').
        interval (int): 캔들 간격 (분 단위, 예: 1, 10, 60).
        start_time (datetime): 수집 시작 시간.
        end_time (datetime): 수집 종료 시간.

    Returns:
        pd.DataFrame: 캔들 데이터.
    """
    base_url = f"https://api.upbit.com/v1/candles/minutes/{interval}"
    headers = {"Accept": "application/json"}
    results = []

    while start_time < end_time:
        query_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "market": market,
            "to": query_time,
            "count": 200  # Upbit API는 한 번에 최대 200개까지 반환
        }

        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code == 429:  # Too Many Requests
                print("429 Too Many Requests. Retrying...")
                time.sleep(1)  # 재시도 전 대기
                retry_count += 1
            else:
                break

        if retry_count == max_retries:
            print(f"Max retries exceeded for {market} at {query_time}. Skipping...")
            continue

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
                "volume": item["candle_acc_trade_volume"],  # 누적거래량
                "value": item["candle_acc_trade_price"]    # 누적 거래 금액
            })

        # 마지막 데이터의 시간으로 이동
        end_time = datetime.strptime(data[-1]["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S") - timedelta(seconds=1)

        # Remaining-Req 헤더를 확인하여 요청 속도 조절
        remaining_req = response.headers.get("Remaining-Req")
        if remaining_req:
            try:
                sec_limit = int(remaining_req.split("sec=")[1])
                if sec_limit > 0:
                    time.sleep(max(1 / sec_limit, 0.1))  # 요청 간 최소 간격 유지
                else:
                    print("Warning: sec_limit is 0. Using default sleep time.")
                    time.sleep(0.5)  # 기본 대기 시간
            except (IndexError, ValueError):
                print("Error parsing Remaining-Req header. Using default sleep time.")
                time.sleep(0.5)  # 기본 대기 시간
        else:
            print("Remaining-Req header not found. Using default sleep time.")
            time.sleep(0.5)  # 기본 대기 시간

    return pd.DataFrame(results)

# 데이터 저장 폴더 생성
output_dir = "F:\\work space\\coin\\price_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 현재 시간 기준으로 60시간 전 데이터 수집
end_time = datetime.now()
start_time = end_time - timedelta(hours=60)

# KRW-BTC, KRW-ETH만 선택
selected_tickers = ["KRW-BTC", "KRW-ETH"]

for ticker in selected_tickers:
    print(f"Fetching data for {ticker} from {start_time} to {end_time}...")
    df = fetch_upbit_candles(ticker, 60, start_time, end_time)

    if not df.empty:
        file_name = f"{output_dir}/{ticker}_last_60_hours.csv"
        df.to_csv(file_name, index=False)
        print(f"{ticker} 데이터 저장 완료: {file_name}")
    else:
        print(f"{ticker} 데이터가 없습니다.")

print("모든 데이터 수집 및 저장이 완료되었습니다.")
