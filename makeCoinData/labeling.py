import pandas as pd
import numpy as np
import os

def calculate_trend(df, window=20):
    """단기 추세 계산"""
    return df['close'].rolling(window=window).mean().diff() > 0

def calculate_volume_indicators(df, window=20):
    """거래량 관련 지표 계산"""
    # 거래량 이동평균
    volume_ma = df['volume'].rolling(window=window).mean()
    
    # 거래량 증가율
    volume_change = df['volume'].pct_change()
    
    # 거래량 강도 (현재 거래량 / 이동평균 거래량)
    volume_strength = df['volume'] / volume_ma
    
    # 가격 변동과 거래량의 관계
    price_volume_correlation = (df['PriceChange'] * volume_change).rolling(window=window).mean()
    
    return volume_ma, volume_change, volume_strength, price_volume_correlation

def calculate_signal_strength(df):
    """각 지표별 신호 강도 계산"""
    signal_strength = pd.DataFrame(index=df.index)
    
    # RSI 신호 강도 (-1 ~ 1)
    signal_strength['rsi_strength'] = (50 - df['RSI']) / 50
    
    # MACD 신호 강도
    signal_strength['macd_strength'] = (df['MACD'] - df['Signal']) / df['close'].abs()
    
    # 볼린저 밴드 신호 강도
    bb_middle = (df['UpperBand'] + df['LowerBand']) / 2
    bb_height = df['UpperBand'] - df['LowerBand']
    signal_strength['bb_strength'] = (bb_middle - df['close']) / (bb_height/2)
    
    return signal_strength

def calculate_volume_weight(df, volume_ma, volume_strength, price_volume_correlation):
    """거래량 기반 가중치 계산"""
    # 거래량이 이동평균보다 높고, 가격-거래량 상관관계가 강할수록 가중치 증가
    volume_percentile = volume_strength.rank(pct=True)
    correlation_weight = (price_volume_correlation + 1) / 2  # -1~1 범위를 0~1로 변환
    
    # 거래량 가중치 계산 (거래량 백분위와 상관관계의 조합)
    volume_weight = (volume_percentile * 0.7 + correlation_weight * 0.3)
    
    return volume_weight

def calculate_volatility_weight(df):
    """변동성 기반 가중치 계산"""
    vol_percentile = df['Volatility'].rank(pct=True)
    return 1 - vol_percentile

def label_data(df):
    # 기본값 설정
    df['label'] = 'hold'
    
    # 추세 계산
    df['trend'] = calculate_trend(df)
    
    # 거래량 지표 계산
    volume_ma, volume_change, volume_strength, price_volume_correlation = calculate_volume_indicators(df)
    
    # 거래량 가중치 계산
    volume_weight = calculate_volume_weight(df, volume_ma, volume_strength, price_volume_correlation)
    
    # 변동성 가중치 계산
    volatility_weight = calculate_volatility_weight(df)
    
    # 신호 강도 계산
    signal_strength = calculate_signal_strength(df)
    
    # 종합 신호 강도 계산 (거래량 가중치 반영)
    combined_strength = (
        signal_strength['rsi_strength'] * 0.3 +
        signal_strength['macd_strength'] * 0.4 +
        signal_strength['bb_strength'] * 0.3
    ) * volume_weight * volatility_weight
    
    # 거래량 기반 임계값 설정
    base_threshold = 0.2
    volume_adjusted_threshold = base_threshold * (1 + (1 - volume_weight))
    
    # 추세와 거래량을 고려한 신호 생성
    df.loc[(combined_strength > volume_adjusted_threshold) & 
           (df['trend']) & 
           (volume_strength > 1), 'label'] = 'buy'
    
    df.loc[(combined_strength < -volume_adjusted_threshold) & 
           (~df['trend']) & 
           (volume_strength > 1), 'label'] = 'sell'
    
    # 극단적인 상황에서의 반전 신호 (높은 거래량 동반 필요)
    extreme_threshold = 0.4
    df.loc[(combined_strength > extreme_threshold) & 
           (volume_strength > 1.5), 'label'] = 'buy'
    
    df.loc[(combined_strength < -extreme_threshold) & 
           (volume_strength > 1.5), 'label'] = 'sell'
    
    # 추가 지표 저장
    df['signal_strength'] = combined_strength
    df['volume_strength'] = volume_strength
    df['volume_ma'] = volume_ma
    df['price_volume_correlation'] = price_volume_correlation
    
    return df

def process_files():
    input_folder = "F:/work space/coin/price_data/techData"
    output_folder = "F:/work space/coin/price_data/label"
    os.makedirs(output_folder, exist_ok=True)
    
    for year in range(2019, 2026):
        input_file = os.path.join(input_folder, f"krw_coin_prices_with_tech_{year}.csv")
        output_file = os.path.join(output_folder, f"krw_coin_prices_with_label_{year}.csv")
        
        if os.path.exists(input_file):
            try:
                print(f"Processing {year} data...")
                df = pd.read_csv(input_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # ticker별로 처리
                processed_dfs = []
                for ticker in df['ticker'].unique():
                    ticker_df = df[df['ticker'] == ticker].copy()
                    ticker_df = ticker_df.sort_values('timestamp')
                    ticker_df = label_data(ticker_df)
                    processed_dfs.append(ticker_df)
                
                final_df = pd.concat(processed_dfs, ignore_index=True)
                final_df.to_csv(output_file, index=False)
                print(f"Successfully processed {year} data")
                
            except Exception as e:
                print(f"Error processing {year} data: {e}")
        else:
            print(f"File {input_file} does not exist.")

if __name__ == "__main__":
    process_files()