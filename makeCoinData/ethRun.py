import pickle
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

class MetaModelPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.meta_model = None
        self.rf_model = None
        self.lgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.target_scaler = None
        self.features = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'Signal', 'Volatility', 'PriceChange']

    def load_models(self):
        try:
            print("Loading models...")
            # 각 모델 파일 경로
            meta_path = f"{self.model_dir}/meta_model_0.0000.pkl"
            rf_path = f"{self.model_dir}/rf_model.pkl"
            lgb_path = f"{self.model_dir}/lgb_model.pkl"
            lstm_path = f"{self.model_dir}/lstm_model"
            scaler_path = f"{self.model_dir}/scaler.pkl"
            target_scaler_path = f"{self.model_dir}/target_scaler.pkl"

            # 모델 로드
            self.meta_model = joblib.load(meta_path)
            self.rf_model = joblib.load(rf_path)
            self.lgb_model = joblib.load(lgb_path)
            self.lstm_model = tf.keras.models.load_model(lstm_path)
            self.scaler = joblib.load(scaler_path)
            self.target_scaler = joblib.load(target_scaler_path)
            
            print("All models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def prepare_data(self, data):
        # 특성 선택
        features_data = data[self.features].values
        
        # 스케일링 적용
        scaled_data = self.scaler.transform(features_data)
        
        # LSTM을 위한 3D 형태로 변환 (samples, time_steps, features)
        lstm_data = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])
        print("LSTM input:", lstm_data[:5])

        # LightGBM과 XGBoost를 위한 2D 형태로 변환
        lgb_rf_data = scaled_data.reshape(1, -1)
        
        return lstm_data, lgb_rf_data

    def predict(self, data):
        if not all([self.meta_model, self.rf_model, self.lgb_model, self.lstm_model]):
            if not self.load_models():
                return None

        try:
            # 데이터 준비
            lstm_data, lgb_rf_data = self.prepare_data(data)

            # 각 모델별 예측
            lstm_pred = self.lstm_model.predict(lstm_data, verbose=0)
            rf_pred = self.rf_model.predict(lgb_rf_data)
            lgb_pred = self.lgb_model.predict(lgb_rf_data)

            # 메타모델 입력을 위한 예측값 결합
            meta_input = np.column_stack((rf_pred, lgb_pred, lstm_pred))

            # 메타모델 최종 예측
            final_prediction = self.meta_model.predict(meta_input)

            # 스케일링 역변환
            final_prediction = self.target_scaler.inverse_transform(final_prediction.reshape(-1, 1))

            return {
                'final_prediction': final_prediction[0][0],
                'individual_predictions': {
                    'lstm': self.target_scaler.inverse_transform(lstm_pred.reshape(-1, 1))[0][0],
                    'rf': self.target_scaler.inverse_transform(rf_pred.reshape(-1, 1))[0][0],
                    'lgb': self.target_scaler.inverse_transform(lgb_pred.reshape(-1, 1))[0][0]
                }
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

def load_recent_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
        recent_data = data.sort_values(by='timestamp', ascending=False).head(60)
        recent_data = recent_data.sort_values(by='timestamp', ascending=True)
        return recent_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    # 파일 경로 설정
    model_dir = r"F:\work space\coin\price_data\models\KRW-ETH_20250114_104017"
    data_file_path = r"F:\work space\coin\price_data\KRW-ETH_with_tech_indicators.csv"
    
    # 예측기 초기화
    predictor = MetaModelPredictor(model_dir)
    
    print("Loading data...")
    recent_data = load_recent_data(data_file_path)
    
    if recent_data is not None:
        print("Making predictions...")
        result = predictor.predict(recent_data)
        
        if result:
            print("\n" + "="*50)
            print("Individual Model Predictions:")
            print(f"LSTM Model: {result['individual_predictions']['lstm']:,.0f} KRW")
            print(f"Random Forest: {result['individual_predictions']['rf']:,.0f} KRW")
            print(f"LightGBM: {result['individual_predictions']['lgb']:,.0f} KRW")
            print("\nMeta Model Final Prediction:")
            print(f"Final Prediction: {result['final_prediction']:,.0f} KRW")
            print("="*50)

if __name__ == "__main__":
    main()