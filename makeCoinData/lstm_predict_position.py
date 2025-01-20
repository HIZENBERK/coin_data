import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import os

# 데이터 로드
def load_data(directory, start_year, end_year):
    file_pattern = f"{directory}/krw_coin_prices_with_label_{{year}}.csv"
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = file_pattern.format(year=year)
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# 데이터 전처리
def preprocess_data(df, window_size=60):
    df = df.sort_values('timestamp')
    features = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    target = df['label'].apply(lambda x: 1 if x == 'buy' else 0 if x == 'sell' else -1)
    target = target.shift(-window_size).dropna().values  # 예측할 타겟값
    scaled_features = scaled_features[:-window_size]

    X, y = [], []
    for i in range(len(scaled_features) - window_size):
        X.append(scaled_features[i:i + window_size])
        y.append(target[i])
    return np.array(X), np.array(y)

# 모델 정의
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 학습 및 평가
def train_and_evaluate(X_train, y_train, X_val, y_val):
    model = build_lstm_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
    return model

# 메인 함수
def main():
    data_dir = "F:/work space/coin/price_data/label"
    start_year, end_year = 2020, 2025
    df = load_data(data_dir, start_year, end_year)

    # 티커별 데이터 처리
    tickers = df['ticker'].unique()
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker]
        X, y = preprocess_data(ticker_df)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 모델 학습 및 저장
        model = train_and_evaluate(X_train, y_train, X_val, y_val)
        model.save(f"F:/work space/coin/price_data/models/{ticker}_lstm_model.h5")

if __name__ == "__main__":
    main()
