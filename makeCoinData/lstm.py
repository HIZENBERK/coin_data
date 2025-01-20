import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras_tuner import RandomSearch
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TerminateOnNaN
from keras.losses import Huber
from sklearn.preprocessing import StandardScaler
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.constraints import MaxNorm


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 데이터 로드
def load_multiple_years_data(start_year, end_year, directory):
    file_pattern = f"{directory}/krw_coin_prices_with_label_{{year}}.csv"
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = file_pattern.format(year=year)
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        df['year'] = year
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# 데이터 전처리
def preprocess_ticker_data(df, target_tickers, window_size=60):
    ticker_data = {}
    for ticker in target_tickers:
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values(by='timestamp')
        #print(ticker_df)
        ticker_df.to_csv(f"F:\work space\coin\price_data\df\{ticker}_df.csv", index=False)
        X, y, scaler, target_scaler = create_sequences(ticker_df, window_size)
        ticker_data[ticker] = (X, y, scaler, target_scaler)
    return ticker_data

# 시퀀스 생성
def create_sequences(data, window_size):
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

def visualize_yearly_data(df, ticker, save_dir="F:/work space/coin/price_data/visualizations"):
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Convert timestamp to datetime if it's not already
    try:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting timestamp: {e}")
        print("Timestamp sample:", df['timestamp'].iloc[0])
        return
    
    # Group data by year
    df['year'] = df['datetime'].dt.year
    years = df['year'].unique()
    
    for year in years:
        year_data = df[df['year'] == year].copy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Price plot
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(year_data['datetime'], year_data['close'], label='Close Price', color='black')
        ax1.set_title(f'{ticker} Price Movement - {year}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # RSI and MACD plot
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(year_data['datetime'], year_data['RSI'], label='RSI', color='blue')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='r', linestyle='--', alpha=0.5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(year_data['datetime'], year_data['MACD'], label='MACD', color='green')
        ax2_twin.plot(year_data['datetime'], year_data['Signal'], label='Signal', color='red')
        
        # Add legends for both y-axes
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.set_title('Technical Indicators - RSI and MACD')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2_twin.set_ylabel('MACD')
        ax2.grid(True)
        
        # Volatility and Price Change plot
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(year_data['datetime'], year_data['Volatility'], label='Volatility', color='purple')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(year_data['datetime'], year_data['PriceChange'], label='Price Change %', color='orange')
        
        # Add legends for both y-axes
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax3.set_title('Volatility and Price Change')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility')
        ax3_twin.set_ylabel('Price Change %')
        ax3.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = f"{save_dir}/{ticker}_{year}_analysis.png"
        plt.savefig(save_path)
        plt.show()
        plt.close()
        
        print(f"Saved visualization for {ticker} - {year} to {save_path}")

def print_data_info(df):
    """데이터 정보를 출력하는 헬퍼 함수"""
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nSample of timestamp values:", df['timestamp'].head())
    print("\nData Types:")
    print(df.dtypes)

# LSTM 모델 정의
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

def tune_and_train_lstm(X_train, y_train, X_val, y_val):
    # Random Search with updated parameters
    tuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=3,  # Reduced trials for faster tuning
        executions_per_trial=3,  # Stability through averaging
        directory='./model_tuning',
        project_name='lstm_tuning_v2'
    )

    # Updated Callbacks
    callbacks = [
        TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # Reduced sensitivity to loss changes
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Hyperparameter search
    tuner.search(
        X_train,
        y_train,
        epochs=5,  # Reduced epochs for tuning speed
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Retrieve the best hyperparameters and train the final model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(
        X_train,
        y_train,
        epochs=10,  # Increased epochs for final training
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return model

# 모델 저장
def save_lstm_model(lstm_model, scaler, target_scaler, mse, ticker, directory="F:/work space/coin/price_data/models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{directory}/{ticker}_{timestamp}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 모델 저장
    model_path = f"{base_dir}/only_lstm_model"
    lstm_model.save(model_path)
    
    # 스케일러 저장
    np.save(f"{base_dir}/scaler.npy", [scaler.mean_, scaler.scale_])
    np.save(f"{base_dir}/target_scaler.npy", [target_scaler.mean_, target_scaler.scale_])
    
    # 모델 정보 저장
    model_info = {
        'lstm_model_path': model_path,
        'scaler_path': f"{base_dir}/scaler.npy",
        'target_scaler_path': f"{base_dir}/target_scaler.npy",
        'mse': float(mse)
    }
    
    with open(f"{base_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print(f"Model and related files saved to {base_dir}")
    return model_info

def save_training_data(X_train, y_train, ticker, directory="F:/work space/coin/price_data/training_data"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # X_train을 2D로 재구성
    # (samples, time_steps, features) -> (samples, time_steps * features)
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_2d = X_train.reshape(n_samples, -1)
    
    # 컬럼 이름 생성
    feature_names = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']
    columns = []
    for t in range(n_timesteps):
        for f in feature_names:
            columns.append(f'{f}_t-{n_timesteps-t}')
    
    # DataFrame 생성 및 저장
    df_train = pd.DataFrame(X_train_2d, columns=columns)
    df_train['target'] = y_train
    
    # CSV 파일로 저장
    file_path = f"{directory}/{ticker}_train_data_{timestamp}.csv"
    df_train.to_csv(file_path, index=False)
    print(f"Training data saved to {file_path}")
    
    return file_path

def check_nan_values(X_train):
    """
    Check and print information about NaN values in the training data
    
    Args:
        X_train: numpy array of shape (samples, timesteps, features)
        
    Returns:
        total_nans: Total number of NaN values found
    """
    # Get total number of NaN values
    total_nans = np.isnan(X_train).sum()
    
    # If there are any NaN values, get detailed information
    if total_nans > 0:
        # Get indices where NaN values exist
        nan_indices = np.where(np.isnan(X_train))
        
        print(f"\nTotal number of NaN values: {total_nans}")
        print(f"Number of samples affected: {len(np.unique(nan_indices[0]))}")
        print("\nNaN locations:")
        print(f"Sample indices: {np.unique(nan_indices[0])}")
        print(f"Timestep indices: {np.unique(nan_indices[1])}")
        print(f"Feature indices: {np.unique(nan_indices[2])}")
        
        # Print distribution of NaN values across features
        feature_nans = np.isnan(X_train).sum(axis=(0,1))
        print("\nNaN count per feature:")
        features = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']
        for idx, count in enumerate(feature_nans):
            if count > 0:
                print(f"{features[idx]}: {count} NaN values")
    else:
        print("\nNo NaN values found in the training data")
    
    return total_nans

def save_ticker_data(ticker_data, directory="F:/work space/coin/price_data/ticker_data"):
    """
    Save ticker_data dictionary to CSV files
    
    Args:
        ticker_data: Dictionary containing (X, y, scaler, target_scaler) for each ticker
        directory: Directory to save the data
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{directory}/{timestamp}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    metadata = {
        'timestamp': timestamp,
        'tickers': list(ticker_data.keys()),
        'data_shapes': {}
    }
    
    for ticker, (X, y, scaler, target_scaler) in ticker_data.items():
        # X 데이터 처리
        n_samples, n_timesteps, n_features = X.shape
        X_2d = X.reshape(n_samples, -1)  # 3D를 2D로 변환
        
        # 컬럼 이름 생성
        features = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']
        columns = []
        for t in range(n_timesteps):
            for f in features:
                columns.append(f'{f}_t-{n_timesteps-t}')
        
        # DataFrame 생성
        df = pd.DataFrame(X_2d, columns=columns)
        df['target'] = y
        
        # 스케일러 정보 저장
        scaler_info = {
            'feature_means': scaler.mean_.tolist(),
            'feature_scales': scaler.scale_.tolist(),
            'target_mean': float(target_scaler.mean_),
            'target_scale': float(target_scaler.scale_)
        }
        
        # CSV 및 스케일러 정보 저장
        df.to_csv(f"{base_dir}/{ticker}_data.csv", index=False)
        with open(f"{base_dir}/{ticker}_scaler_info.json", 'w') as f:
            json.dump(scaler_info, f, indent=4)
        
        metadata['data_shapes'][ticker] = {
            'n_samples': n_samples,
            'n_timesteps': n_timesteps,
            'n_features': n_features
        }
    
    # 메타데이터 저장
    with open(f"{base_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Ticker data saved to {base_dir}")
    return base_dir

# 메인 실행 코드
def main():
    # 데이터 로드
    start_year, end_year = 2020, 2024
    data_dir = "F:/work space/coin/price_data/label"
    df = load_multiple_years_data(start_year, end_year, data_dir)

    # 학습할 티커 선택
    selected_tickers = ['KRW-BTC', 'KRW-ETH']
    ticker_data = preprocess_ticker_data(df, selected_tickers)
    # 데이터 정보 출력
    #print_data_info(df)
    for ticker in selected_tickers:
        print(f"\nProcessing visualization for {ticker}")
        ticker_df = df[df['ticker'] == ticker].copy()
        visualize_yearly_data(ticker_df, ticker)
    #print(ticker_data)
    #saved_dir = save_ticker_data(ticker_data)
    #print(f"Saved ticker data to: {saved_dir}")
    for ticker, (X, y, scaler, target_scaler) in ticker_data.items():
        print(f"\nProcessing ticker: {ticker}")
        
        # 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # NaN 값 체크
        nan_count = check_nan_values(X_train)
        if nan_count > 0:
            print("Warning: NaN values detected in training data!")
        # 학습 데이터 저장
        #save_training_data(X_train, y_train, ticker)
        
        # LSTM 모델 학습
        lstm_model = tune_and_train_lstm(X_train, y_train, X_val, y_val)
        
        if lstm_model is not None:
            # 모델 평가
            mse = lstm_model.evaluate(X_val, y_val, verbose=0)
            print(f"{ticker} - Mean Squared Error: {mse}")
            
            # 모델 저장
            save_lstm_model(lstm_model, scaler, target_scaler, mse, ticker)

if __name__ == "__main__":
    main()