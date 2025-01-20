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
import os
import json
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import KFold

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
def build_lstm_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=64, step=32),
        input_shape=(60, 9),
        return_sequences=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        kernel_constraint=MaxNorm(3),
        recurrent_constraint=MaxNorm(3),
        bias_constraint=MaxNorm(3)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout1', [0.3, 0.5, 0.7])))

    model.add(LSTM(
        units=hp.Int('units_2', min_value=32, max_value=64, step=32),
        return_sequences=False,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        kernel_constraint=MaxNorm(3),
        recurrent_constraint=MaxNorm(3),
        bias_constraint=MaxNorm(3)
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout2', [0.3, 0.5, 0.7])))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('dropout3', [0.3, 0.5, 0.7])))

    model.add(Dense(1))

    optimizer = Adam(
        learning_rate=hp.Choice('lr', [1e-4, 1e-3])
    )

    model.compile(
        optimizer=optimizer,
        loss=Huber(delta=1.0)
    )
    return model

def tune_lstm_model(X_train, y_train):
    # Random Search with updated parameters
    tuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=3,
        executions_per_trial=3,
        directory='./model_tuning',
        project_name='lstm_tuning_v2'
    )

    # Updated Callbacks
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    try:
        tuner.search(
            X_train, 
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        print(f"Tuning failed: {e}")
        return None, None

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    
    # Final training with best hyperparameters
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, best_hps

def objective_lgb(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 1e-8, 1.0, log=True)
    }
    
    # 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # callbacks 설정
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    # 모델 학습
    model = lgb.train(
        param,
        train_data,
        valid_sets=[valid_data],
        callbacks=callbacks
    )
    
    # 검증 데이터에 대한 예측
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    
    return mse
"""
def objective_xgb(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        
        # 트리 구조 관련 파라미터
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        
        # 학습 관련 파라미터
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
        
        # 정규화 관련 파라미터
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
    }
    
    # Cross-validation을 통한 안정적인 성능 평가
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in kf.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[valid_idx]
        y_t, y_v = y_train[train_idx], y_train[valid_idx]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_t, y_t,
                 eval_set=[(X_v, y_v)],
                 callbacks=[xgb.callback.EarlyStopping(rounds=50)],
                 verbose=False)
        
        pred = model.predict(X_v)
        score = mean_squared_error(y_v, pred)
        scores.append(score)
    
    return np.mean(scores)
"""
def objective_xgb(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        "device" : "cuda",
        #'gpu_id': 0,
        
        # 트리 구조 관련 파라미터
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        
        # 학습 관련 파라미터
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        
        # 정규화 관련 파라미터
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in kf.split(X_train):
        X_t, X_v = X_train[train_idx], X_train[valid_idx]
        y_t, y_v = y_train[train_idx], y_train[valid_idx]
        

        model = xgb.XGBRegressor(**param)
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False
        )
        
        pred = model.predict(X_v)
        score = mean_squared_error(y_v, pred)
        scores.append(score)
    
    return np.mean(scores)

def optimize_models(X_train, y_train, X_val, y_val, n_trials=50):
    # XGBoost 최적화
    print("\nOptimizing XGBoost...")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val), 
                      n_trials=n_trials)
    
    # LightGBM 최적화
    print("Optimizing LightGBM...")
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_train, y_train, X_val, y_val), 
                      n_trials=n_trials)
    
    print("\nBest LightGBM parameters:", study_lgb.best_params)
    print("Best XGBoost parameters:", study_xgb.best_params)
    
    return study_lgb.best_params, study_xgb.best_params

# LightGBM 및 XGBoost 모델 학습
def train_models(X_train, y_train, X_val, y_val):
    # 2차원 배열로 변환
    X_train_lgb_rf, X_val_lgb_rf = reshape_for_lgb_rf(X_train, X_val)
    
    # LightGBM과 XGBoost 파라미터 최적화
    lgb_params, xgb_params = optimize_models(X_train_lgb_rf, y_train, X_val_lgb_rf, y_val)
    
    # XGBoost 모델 학습
    print("Training XGBoost with optimal parameters...")
    rf_model = xgb.XGBRegressor(**xgb_params)
    rf_model.fit(X_train_lgb_rf, y_train,
                eval_set=[(X_val_lgb_rf, y_val)],
                early_stopping_rounds=50,
                verbose=False)
    rf_pred = rf_model.predict(X_val_lgb_rf)
    
    # LightGBM 모델 학습
    print("\nTraining LightGBM with optimal parameters...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train_lgb_rf, y_train,
                 eval_set=[(X_val_lgb_rf, y_val)],
                 early_stopping_rounds=50,
                 verbose=False)
    lgb_pred = lgb_model.predict(X_val_lgb_rf)
    
    print("Training LSTM...")
    lstm_model, best_hps = tune_lstm_model(X_train, y_train)
    
    if lstm_model is None:
        print("LSTM model training failed")
        return None, None, None, None, None, None
        
    lstm_pred = lstm_model.predict(X_val)
    
    return rf_model, lgb_model, lstm_model, rf_pred, lgb_pred, lstm_pred

# Stacking 모델
def stack_models(rf_pred, lgb_pred, lstm_pred, y_val):
    print("Training Meta Model (RandomForest)...")
    stacked_preds = np.column_stack((rf_pred, lgb_pred, lstm_pred))
    meta_model = RandomForestRegressor(n_estimators=100)
    meta_model.fit(stacked_preds, y_val)
    return meta_model

# 모델 저장
def save_models(meta_model, rf_model, lgb_model, lstm_model, scaler, target_scaler, score, ticker, directory="F:/work space/coin/price_data/models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 메인 디렉토리 생성
    base_dir = f"{directory}/{ticker}_{timestamp}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 메타 모델 저장
    meta_model_path = f"{base_dir}/meta_model_{score:.4f}.pkl"
    joblib.dump(meta_model, meta_model_path)
    
    # 기본 모델들 저장
    joblib.dump(rf_model, f"{base_dir}/rf_model.pkl")
    joblib.dump(lgb_model, f"{base_dir}/lgb_model.pkl")
    lstm_model.save(f"{base_dir}/lstm_model")  # LSTM은 .save() 메소드 사용
    
    # 스케일러 저장
    joblib.dump(scaler, f"{base_dir}/scaler.pkl")
    joblib.dump(target_scaler, f"{base_dir}/target_scaler.pkl")
    
    print(f"All models for {ticker} saved to {base_dir}")
    
    # 모델 경로 정보를 담은 딕셔너리 반환
    model_paths = {
        'meta_model': meta_model_path,
        'rf_model': f"{base_dir}/rf_model.pkl",
        'lgb_model': f"{base_dir}/lgb_model.pkl",
        'lstm_model': f"{base_dir}/lstm_model",
        'scaler': f"{base_dir}/scaler.pkl",
        'target_scaler': f"{base_dir}/target_scaler.pkl"
    }
    
    # 경로 정보를 JSON 파일로 저장
    with open(f"{base_dir}/model_paths.json", 'w') as f:
        json.dump(model_paths, f, indent=4)
    
    return model_paths

# 데이터 로드
start_year, end_year = 2020, 2025
data_dir = "F:/work space/coin/price_data/label"
df = load_multiple_years_data(start_year, end_year, data_dir)

# 특정 ticker만 학습
selected_tickers = ['KRW-BTC', 'KRW-ETH']
ticker_data = preprocess_ticker_data(df, selected_tickers)
all_models_info = {}

for ticker, (X, y, scaler, target_scaler) in ticker_data.items():
    print(f"\nProcessing ticker: {ticker}")
    
    # 학습 데이터와 검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 모델 학습
    rf_model, lgb_model, lstm_model, rf_pred, lgb_pred, lstm_pred = train_models(X_train, y_train, X_val, y_val)
    
    if rf_model is None or lgb_model is None or lstm_model is None:
        print(f"Skipping {ticker} due to failed model training")
        continue
    
    # Stacking 모델 학습
    meta_model = stack_models(rf_pred, lgb_pred, lstm_pred, y_val)
    
    # Stacked 예측 및 성능 평가
    stacked_preds = np.column_stack((rf_pred, lgb_pred, lstm_pred))
    final_preds = meta_model.predict(stacked_preds)
    mse = mean_squared_error(y_val, final_preds)
    print(f"{ticker} - Mean Squared Error: {mse}")
    
    # 모델 및 관련 정보 저장
    model_info = save_models(
        meta_model=meta_model,
        rf_model=rf_model,
        lgb_model=lgb_model,
        lstm_model=lstm_model,
        scaler=scaler,
        target_scaler=target_scaler,
        score=mse,
        best_hps=lstm_model.get_config(),  # LSTM 설정 저장
        lgb_params=lgb_model.get_params(),  # LightGBM 파라미터 저장
        xgb_params=rf_model.get_params(),   # XGBoost 파라미터 저장
        ticker=ticker
    )
    
    # 모델 정보 저장
    all_models_info[ticker] = model_info

# 전체 모델 정보를 JSON 파일로 저장
output_dir = "F:/work space/coin/price_data/models"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"{output_dir}/all_models_info_{timestamp}.json", 'w', encoding='utf-8') as f:
    json.dump(all_models_info, f, indent=4, ensure_ascii=False)

print("\nAll models have been trained and saved successfully!")