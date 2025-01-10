import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import joblib
from datetime import datetime
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 데이터 전처리 (예시)
def preprocess_data(df):
    # 필요한 특성 선택
    X = df[['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']]
    y = df[['high', 'low']]  # 예측하려는 값 (high, low 평균)
    return X, y

# LSTM 모델 정의
def create_lstm_model(input_shape, units=50, activation='relu', batch_size=32, epochs=10):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=input_shape))
    model.add(Dense(2))  # high와 low 예측
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# 모델 학습 및 예측
def train_models(X_train, y_train, X_val, y_val):
    print("Training RandomForest...")
    # RandomForest Hyperparameter tuning with GridSearchCV
    rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
    rf_model = GridSearchCV(RandomForestRegressor(), rf_params, cv=3, n_jobs=-1, verbose=1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    print(f"Best RandomForest params: {rf_model.best_params_}")

    print("Training LightGBM...")
    # LightGBM Hyperparameter tuning with GridSearchCV
    lgb_params = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [10, 20, -1]}
    lgb_model = GridSearchCV(lgb.LGBMRegressor(boosting_type='gpu_hist', device='gpu'), lgb_params, cv=3, n_jobs=-1, verbose=1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    print(f"Best LightGBM params: {lgb_model.best_params_}")

    print("Training LSTM...")
    # Ensure TensorFlow uses GPU
    with tf.device('/GPU:0'):
        lstm_model = create_lstm_model((X_train.shape[1], 1), units=50, activation='relu', epochs=10, batch_size=32)
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    lstm_pred = lstm_model.predict(X_val)

    return rf_pred, lgb_pred, lstm_pred

# Stacking을 위한 메타 모델 학습
def stack_models(rf_pred, lgb_pred, lstm_pred, y_val):
    print("Training Meta Model (RandomForest)...")
    stacked_preds = np.column_stack((rf_pred, lgb_pred, lstm_pred))
    meta_model = RandomForestRegressor(n_estimators=100)
    meta_model.fit(stacked_preds, y_val)
    return meta_model

def load_multiple_years_data(start_year, end_year, directory):
    # 연도별 파일 패턴 생성
    file_pattern = f"{directory}/krw_coin_prices_with_label_{{year}}.csv"
    
    # 각 연도에 대해 데이터 로드 후 결합
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = file_pattern.format(year=year)
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df['year'] = year  # 연도 컬럼 추가 (선택사항)
        dfs.append(df)
    
    # 데이터프레임 결합
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# 모델 저장 함수
def save_model(model, score, directory="F:/work space/coin/price_data"):
    # 생성시간과 평가점수를 포함한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"model_{timestamp}_{score:.4f}.pkl"
    file_path = f"{directory}/{file_name}"
    
    # 모델 저장
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

# 데이터 로드 및 전처리
df = load_multiple_years_data(2019, 2025, "F:/work space/coin/price_data/label")
X, y = preprocess_data(df)

# 학습, 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 모델 학습 및 예측
rf_pred, lgb_pred, lstm_pred = train_models(X_train, y_train, X_val, y_val)

# Stacking 메타 모델 학습
meta_model = stack_models(rf_pred, lgb_pred, lstm_pred, y_val)

# 예측 및 평가
stacked_preds = np.column_stack((rf_pred, lgb_pred, lstm_pred))
final_preds = meta_model.predict(stacked_preds)

# 평가 지표: 평균가 예측 범위 내에 실제 평균가가 들어가지 않으면 감점
y_val_avg = (y_val['high'] + y_val['low']) / 2
final_avg_preds = (final_preds[:, 0] + final_preds[:, 1]) / 2
mse = mean_squared_error(y_val_avg, final_avg_preds)
print(f'Mean Squared Error: {mse}')

# 모델 저장
save_model(meta_model, mse)
