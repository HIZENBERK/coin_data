import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras_tuner import RandomSearch
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN
from keras.losses import Huber
from sklearn.preprocessing import StandardScaler
import optuna
import xgboost as xgb

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# XGBoost 모델 하이퍼파라미터 최적화 함수
def objective(trial, X_train, y_train, X_val, y_val):
    # Optuna에서 제공하는 하이퍼파라미터 탐색 공간 설정
    n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 500])
    max_depth = trial.suggest_int('max_depth', 10, 30)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.1)
    
    # XGBoost 모델 초기화
    xgb_model_gpu = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        learning_rate=learning_rate,
        tree_method='gpu_hist',  # GPU에서 histogram 방식으로 학습
        gpu_id=0,                # 첫 번째 GPU 사용
        random_state=42
    )

    # 모델 학습
    xgb_model_gpu.fit(X_train, y_train)
    
    # 검증 데이터셋에 대한 예측
    xgb_pred_gpu = xgb_model_gpu.predict(X_val)
    
    # 예측값과 실제 값의 차이를 기반으로 성능 평가 (MSE)
    mse = np.mean((xgb_pred_gpu - y_val) ** 2)
    
    return mse  # MSE가 낮을수록 좋음

# Optuna 최적화 함수
def optimize_xgb_model(X_train, y_train, X_val, y_val):
    study = optuna.create_study(direction='minimize')  # MSE 최소화
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)
    
    # 최적 파라미터 출력
    print("Best hyperparameters: ", study.best_params)
    
    return study.best_params

# 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 여러 연도의 데이터를 로드
def load_multiple_years_data(start_year, end_year, directory):
    file_pattern = f"{directory}/krw_coin_prices_with_label_{{year}}.csv"
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = file_pattern.format(year=year)
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df['year'] = year  # 연도 컬럼 추가 (선택사항)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# 데이터 전처리 (특정 ticker만 사용)
def preprocess_ticker_data(df, target_tickers, window_size=60):
    tickers = target_tickers
    ticker_data = {}

    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values(by='timestamp')
        X, y, scaler, target_scaler = create_sequences(ticker_df, window_size)
        ticker_data[ticker] = (X, y, scaler, target_scaler)

    return ticker_data

# 시퀀스 생성 함수
def create_sequences(data, window_size):
    # StandardScaler 적용
    scaler = StandardScaler()
    features = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']
    scaled_data = scaler.fit_transform(data[features])
    
    target = (data['high'].values + data['low'].values) / 2
    target_scaler = StandardScaler()
    target = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i + window_size])
        y.append(target[i + window_size])
    
    return np.array(X), np.array(y), scaler, target_scaler

# LSTM 모델을 위한 데이터는 그대로 유지하고,
# LightGBM과 RandomForest 모델은 2차원 배열로 변환합니다.
def reshape_for_lgb_rf(X_train, X_val):
    # LSTM 모델의 경우, 3차원 배열 (samples, time_steps, features) 이 필요하므로 그대로 둡니다.
    # LightGBM과 RandomForest는 2차원 배열 (samples, features) 을 필요로 하므로,
    # 'time_steps' 차원 (60)을 제거하고 features만 남깁니다.
    X_train_lgb_rf = X_train.reshape(X_train.shape[0], -1)  # (samples, time_steps * features)
    X_val_lgb_rf = X_val.reshape(X_val.shape[0], -1)  # (samples, time_steps * features)
    return X_train_lgb_rf, X_val_lgb_rf

# LSTM 모델 생성
"""
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units1', 50, 150, step=25), return_sequences=True, input_shape=(60, 9)))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(hp.Int('units2', 50, 150, step=25)))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('lr', [0.00000000000000000001, 0.00000000000005, 0.00000000000000001])), loss='mean_squared_error')
    return model
"""
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        input_shape=(60, 9),
        return_sequences=False,
        # Gradient clipping 추가
        kernel_constraint=tf.keras.constraints.MaxNorm(3),
        recurrent_constraint=tf.keras.constraints.MaxNorm(3)
    ))
    model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    
    # Learning rate 범위 조정
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice('lr', [1e-25, 5e-25, 1e-26])
        ),
        loss=Huber()  # MSE 대신 Huber loss 사용
    )
    return model

# LSTM 최적화
def tune_lstm_model(X_train, y_train):
    print(f"Input shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print("NaN in X_train:", np.isnan(X_train).any())
    print("NaN in y_train:", np.isnan(y_train).any())
    print("Inf in X_train:", np.isinf(X_train).any())
    print("Inf in y_train:", np.isinf(y_train).any())
    tuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=10,
        directory='F:\work space\coin\price_data\models\searching',
        project_name='lstm_tuning'
    )
    try:
        tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=1, callbacks=[TerminateOnNaN()])
    except Exception as e:
        print(f"Tuning failed: {e}")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    return model, best_hps

# LightGBM 및 XGBoost 모델 학습
def train_models(X_train, y_train, X_val, y_val):
    print("Training LSTM...")
    X_train_lstm = np.expand_dims(X_train, axis=-1)
    X_val_lstm = np.expand_dims(X_val, axis=-1)
    lstm_model, best_hps = tune_lstm_model(X_train_lstm, y_train)
    # Early stopping과 learning rate 조정 추가
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        TerminateOnNaN()
    ]
    
    lstm_model.fit(
        X_train, 
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    lstm_pred = lstm_model.predict(X_val_lstm)
    
    print("Training LightGBM...")
    
    # 2차원 배열로 변환
    X_train_lgb_rf, X_val_lgb_rf = reshape_for_lgb_rf(X_train, X_val)
    
    lgb_params = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.0000001, 0.0000005, 0.000001, 0.000002],
        'max_depth': [10, 20, 30, -1],
        'device': ['gpu'],  # GPU 사용 설정을 리스트로 감싸기
        'boosting_type': ['gbdt'],  # GBDT(Gradient Boosting Decision Tree) 사용
    }
    lgb_model = GridSearchCV(lgb.LGBMRegressor(), lgb_params, cv=3, n_jobs=-1, verbose=1)
    lgb_model.fit(X_train_lgb_rf, y_train)
    lgb_pred = lgb_model.predict(X_val_lgb_rf)
    print(f"Best LightGBM params: {lgb_model.best_params_}")

    print("Training XGBoost...")
    best_params = optimize_xgb_model(X_train_lgb_rf, y_train, X_val_lgb_rf, y_val)
    
    # 최적의 하이퍼파라미터로 XGBoost 모델 학습
    rf_model = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_child_weight=best_params['min_child_weight'],
        learning_rate=best_params['learning_rate'],
        tree_method='gpu_hist',  # GPU에서 histogram 방식으로 학습
        gpu_id=0,                # 첫 번째 GPU 사용
        random_state=42
    )
    
    # 모델 학습
    rf_model.fit(X_train_lgb_rf, y_train)
    
    # 검증 데이터셋에 대한 예측
    rf_pred = rf_model.predict(X_val_lgb_rf)
    print(f"Best XGBoost params: {rf_model.get_params()}")

    return rf_model, lgb_model, lstm_model, rf_pred, lgb_pred, lstm_pred

# Stacking 모델
def stack_models(rf_pred, lgb_pred, lstm_pred, y_val):
    print("Training Meta Model (RandomForest)...")
    stacked_preds = np.column_stack((rf_pred, lgb_pred, lstm_pred))
    meta_model = RandomForestRegressor(n_estimators=100)
    meta_model.fit(stacked_preds, y_val)
    return meta_model

# 모델 저장
def save_model(model, score, ticker, directory="F:/work space/coin/price_data/models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"model_{ticker}_{timestamp}_{score:.4f}.pkl"
    file_path = f"{directory}/{file_name}"
    joblib.dump(model, file_path)
    print(f"Model for {ticker} saved to {file_path}")

# 데이터 로드
start_year, end_year = 2019, 2025
data_dir = "F:/work space/coin/price_data/label"
df = load_multiple_years_data(start_year, end_year, data_dir)

# 특정 ticker만 학습
selected_tickers = ['KRW-BTC', 'KRW-ETH']
ticker_data = preprocess_ticker_data(df, selected_tickers)

all_meta_models = {}

for ticker, (X, y, scaler, target_scaler) in ticker_data.items():
    print(f"Processing ticker: {ticker}")
    
    # 학습 데이터와 검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 모델 학습
    rf_model, lgb_model, lstm_model, rf_pred, lgb_pred, lstm_pred = train_models(X_train, y_train, X_val, y_val)
    
    # Stacking 모델 학습
    meta_model = stack_models(rf_pred, lgb_pred, lstm_pred, y_val)
    
    # Stacked 예측
    stacked_preds = np.column_stack((rf_pred, lgb_pred, lstm_pred))
    final_preds = meta_model.predict(stacked_preds)
    
    # 성능 평가
    mse = mean_squared_error(y_val, final_preds)
    print(f"{ticker} - Mean Squared Error: {mse}")
    
    # 모델 저장
    save_model(meta_model, mse, ticker)
    
    # 모든 메타 모델 저장
    all_meta_models[ticker] = meta_model
