import pyupbit
import pandas as pd
from datetime import datetime
import os

# 윤년 판단 함수
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# 2019년부터 현재 연도까지
start_year = 2019
end_year = datetime.now().year

# 모든 KRW 코인 티커 가져오기
krw_tickers = pyupbit.get_tickers(fiat="KRW")

# 데이터프레임 리스트 생성
df_list = []

for year in range(start_year, end_year + 1):
    # 윤년 판단하여 count 설정
    if is_leap_year(year):
        count = 8784  # 윤년: 366일 × 24시간
    else:
        count = 8760  # 평년: 365일 × 24시간

    # 연도별 시작일과 종료일 설정
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 59)

    for ticker in krw_tickers:
        # 각 티커의 데이터 가져오기 (1시간 간격)
        df = pyupbit.get_ohlcv(ticker, interval="minute60", to=end_date, count=count)
        if df is not None and not df.empty:
            # 열 이름에 티커를 포함하기 위해 MultiIndex 사용
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            df_list.append(df)
            print(f"{ticker} {year} 데이터 저장 완료")

# 병합 (모든 데이터 하나로 합치기)
df_final = pd.concat(df_list, axis=1)

# 폴더 확인 및 저장
#F:\\work space\\coin\\price_data
output_dir = 'F:\\work space\\coin\\price_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 데이터 저장
df_final.to_csv(f'{output_dir}\\krw_coin_prices_{start_year}_to_{end_year}.csv', index=True)
print("모든 데이터가 저장되었습니다.")
