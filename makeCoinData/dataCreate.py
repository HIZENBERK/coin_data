import pyupbit
import pandas as pd
from datetime import datetime, timedelta
import os

# 모든 KRW 코인 티커 가져오기
krw_tickers = pyupbit.get_tickers(fiat="KRW")
end_date = datetime.now()

# 데이터프레임 리스트 생성
df_list = []

for ticker in krw_tickers:
    # 각 티커의 데이터 가져오기 (1시간 간격, 365개의 데이터)
    df = pyupbit.get_ohlcv(ticker, interval="minute60", to=end_date, count=8784)
    if df is not None and not df.empty:
        # 열 이름에 티커를 포함하기 위해 MultiIndex 사용
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        df_list.append(df)
        print(f"{ticker} 데이터 저장 완료")

# 병합 (모든 데이터 하나로 합치기)
df_final = pd.concat(df_list, axis=1)

# 폴더 확인 및 저장
output_dir = '..\\price_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 데이터 저장
df_final.to_csv(f'{output_dir}\\krw_coin_prices_1year.csv', index=True)
print("모든 데이터가 저장되었습니다.")
