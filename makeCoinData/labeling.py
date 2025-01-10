import pandas as pd
import os

# 라벨링 함수 정의
def label_data(df):
    # 값이 비어있을 경우 'hold'로 지정
    df['label'] = 'hold'

    # RSI 기준
    df.loc[df['RSI'] <= 30, 'label'] = 'buy'
    df.loc[df['RSI'] >= 70, 'label'] = 'sell'

    # MACD 기준: MACD가 Signal을 상향 돌파하면 buy, 하향 돌파하면 sell
    df.loc[(df['MACD'] > df['Signal']) & (df['RSI'] != 70), 'label'] = 'buy'
    df.loc[(df['MACD'] < df['Signal']) & (df['RSI'] != 30), 'label'] = 'sell'

    # 볼린저 밴드 기준: 가격이 하단 밴드를 돌파하면 buy, 상단 밴드를 돌파하면 sell
    df.loc[df['close'] < df['LowerBand'], 'label'] = 'buy'
    df.loc[df['close'] > df['UpperBand'], 'label'] = 'sell'

    return df

# 파일 경로 설정
input_folder = "F:/work space/coin/price_data/techData"
output_folder = "F:/work space/coin/price_data/label"

# 2020년부터 현재 년도까지 파일을 처리
for year in range(2019, 2026):
    input_file = os.path.join(input_folder, f"krw_coin_prices_with_tech_{year}.csv")
    output_file = os.path.join(output_folder, f"krw_coin_prices_with_label_{year}.csv")

    # 파일이 존재할 경우 처리
    if os.path.exists(input_file):
        # CSV 파일 읽기
        df = pd.read_csv(input_file)

        # 라벨링 적용
        df = label_data(df)

        # 라벨링된 데이터 저장
        df.to_csv(output_file, index=False)

        print(f"Processed {input_file} and saved to {output_file}")
    else:
        print(f"File {input_file} does not exist.")
