import os
import pandas as pd

# 경로 설정
base_path = r'F:\work space\coin\price_data'

# 파일 이름 패턴
file_pattern = "krw_coin_prices_"

# 2019년부터 2025년까지의 파일 처리
for year in range(2019, 2026):
    file_name = f"{file_pattern}{year}.csv"
    file_path = os.path.join(base_path, file_name)

    # 파일이 존재하는지 확인
    if os.path.exists(file_path):
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path)

            # ticker가 ['KRW-BTC', 'KRW-ETH']인 데이터만 필터링
            df_filtered = df[df['ticker'].isin(['KRW-BTC', 'KRW-ETH'])]

            # timestamp 컬럼을 datetime으로 변환
            df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

            # timestamp 기준으로 정렬
            df_sorted = df_filtered.sort_values(by='timestamp', ascending=True)

            # 정렬된 데이터 저장
            df_sorted.to_csv(file_path, index=False)

            print(f"{file_name} 파일 필터링, 정렬 및 저장 완료.")
        except Exception as e:
            print(f"{file_name} 파일 처리 중 오류 발생: {e}")
    else:
        print(f"{file_name} 파일이 경로에 존재하지 않습니다.")
