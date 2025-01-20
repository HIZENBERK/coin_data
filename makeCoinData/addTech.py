import os
import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data['close'].diff(1)
    
    # 상승분과 하락분 계산
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # 첫 평균 계산
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # 그 이후 평균 계산
    for i in range(window, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window-1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window-1) + loss.iloc[i]) / window
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['close'].ewm(span=short_window, adjust=False, min_periods=1).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False, min_periods=1).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False, min_periods=1).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['close'].rolling(window=window, min_periods=1).mean()
    rolling_std = data['close'].rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

def calculate_volatility(data, window=14):
    daily_vol = data['close'].pct_change().rolling(window=window, min_periods=1).std()
    annual_vol = daily_vol * np.sqrt(252)  # 252는 연간 거래일 수
    return annual_vol

def calculate_price_change(data):
    return data['close'].pct_change()

def process_and_save_data():
    # 경로 설정
    base_path = r'F:\work space\coin\price_data'
    output_path = os.path.join(base_path, "techData")
    os.makedirs(output_path, exist_ok=True)

    # 모든 데이터를 저장할 리스트
    all_data = []

    # 2019년부터 2025년까지의 데이터 읽기
    for year in range(2019, 2026):
        file_name = f"krw_coin_prices_{year}.csv"
        file_path = os.path.join(base_path, file_name)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                all_data.append(df)
                print(f"{file_name} 파일 로드 완료")
            except Exception as e:
                print(f"{file_name} 파일 읽기 중 오류 발생: {e}")
        else:
            print(f"{file_name} 파일이 경로에 존재하지 않습니다.")

    if not all_data:
        print("처리할 데이터가 없습니다.")
        return

    # 모든 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # timestamp로 정렬
    combined_df = combined_df.sort_values(['ticker', 'timestamp'])
    
    # 동일 ticker 내에서 중복되는 timestamp 제거
    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'ticker'], keep='last')

    # ticker별로 그룹화하여 기술지표 계산
    processed_dfs = []
    for ticker in combined_df['ticker'].unique():
        print(f"{ticker} 처리 중...")
        ticker_df = combined_df[combined_df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('timestamp')
        
        # 기술지표 계산
        try:
            ticker_df['RSI'] = calculate_rsi(ticker_df)
            ticker_df['MACD'], ticker_df['Signal'] = calculate_macd(ticker_df)
            ticker_df['MiddleBand'], ticker_df['UpperBand'], ticker_df['LowerBand'] = calculate_bollinger_bands(ticker_df)
            ticker_df['Volatility'] = calculate_volatility(ticker_df)
            ticker_df['PriceChange'] = calculate_price_change(ticker_df)
            
            processed_dfs.append(ticker_df)
        except Exception as e:
            print(f"{ticker} 처리 중 오류 발생: {e}")
            continue

    # 처리된 데이터 합치기
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    # 년도별로 분리하여 저장
    for year in range(2019, 2026):
        year_df = final_df[final_df['timestamp'].dt.year == year]
        if not year_df.empty:
            output_file_name = f"krw_coin_prices_with_tech_{year}.csv"
            output_file_path = os.path.join(output_path, output_file_name)
            year_df.to_csv(output_file_path, index=False)
            print(f"{year}년 데이터 저장 완료: {output_file_path}")

if __name__ == "__main__":
    process_and_save_data()