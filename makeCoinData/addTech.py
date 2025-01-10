import os
import pandas as pd
import numpy as np

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

# 경로 설정
base_path = r'F:\work space\coin\price_data'
output_path = os.path.join(base_path, "techData")
os.makedirs(output_path, exist_ok=True)

# 파일 이름 패턴
file_pattern = "krw_coin_prices_"

# 누적 데이터를 담을 DataFrame
cumulative_data = None

# 2019년부터 2025년까지의 파일 처리
for year in range(2019, 2026):
    file_name = f"{file_pattern}{year}.csv"
    file_path = os.path.join(base_path, file_name)

    # 파일이 존재하는지 확인
    if os.path.exists(file_path):
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)

            # timestamp 컬럼을 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 누적 데이터를 이전에 이미 처리한 데이터와 합침
            if cumulative_data is not None:
                # 해당 연도의 데이터는 제외하고, 이전 데이터를 추가
                prev_year_data = cumulative_data[cumulative_data['timestamp'].dt.year == year-1]
                df = pd.concat([prev_year_data, df], ignore_index=True)

            # 해당 연도 데이터에서 이전 연도의 데이터까지 포함시킨 뒤, 새로운 데이터는 처리
            # RSI, MACD, BollingerBands 등 기술 지표를 추가
            df['RSI'] = calculate_rsi(df)
            df['MACD'], df['Signal'] = calculate_macd(df)
            df['MiddleBand'], df['UpperBand'], df['LowerBand'] = calculate_bollinger_bands(df)
            df['Volatility'] = calculate_volatility(df)
            df['PriceChange'] = calculate_price_change(df)

            # 중복 데이터 제거: 동일한 timestamp와 ticker가 중복된 경우만 제거 (ticker가 다르면 모두 살린다)
            df = df.drop_duplicates(subset=['timestamp', 'ticker'], keep='last')

            # 이전 년도의 마지막 데이터부터 새로운 데이터를 합침
            if year > 2019:
                # 현재 연도의 데이터에서 1월 1일 이전에 포함되어야 할 데이터 추출
                df_current_year = df[df['timestamp'].dt.year == year]
                prev_year_data = df[df['timestamp'].dt.year == year-1]
                # 이전 연도의 마지막 데이터부터 새로운 데이터를 합침
                df_current_year = pd.concat([prev_year_data, df_current_year], ignore_index=True)

                # 이전 연도 데이터 저장
                prev_year_file_name = f"krw_coin_prices_with_tech_{year-1}.csv"
                prev_year_file_path = os.path.join(output_path, prev_year_file_name)
                prev_year_data.to_csv(prev_year_file_path, index=False)

            # 현재 연도의 파일만 저장
            output_file_name = f"krw_coin_prices_with_tech_{year}.csv"
            output_file_path = os.path.join(output_path, output_file_name)

            df_current_year.to_csv(output_file_path, index=False)

            print(f"{file_name} 파일 처리 및 저장 완료: {output_file_path}")

            # 누적 데이터 업데이트
            cumulative_data = df.copy()
        except Exception as e:
            print(f"{file_name} 파일 처리 중 오류 발생: {e}")
    else:
        print(f"{file_name} 파일이 경로에 존재하지 않습니다.")
