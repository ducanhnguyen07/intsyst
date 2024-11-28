from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Custom objects for TensorFlow models
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load các mô hình dự đoán điểm thi
models = {
    "cnn": tf.keras.models.load_model("model-b2/cnn_model.h5", custom_objects=custom_objects),
    "dnn": tf.keras.models.load_model("model-b2/dnn_model.h5", custom_objects=custom_objects),
    "rnn": tf.keras.models.load_model("model-b2/rnn_model.h5", custom_objects=custom_objects),
    "lstm": tf.keras.models.load_model("model-b2/lstm_model.h5", custom_objects=custom_objects),
}

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Đăng ký hàm mse như một hàm tùy chỉnh
tf.keras.utils.get_custom_objects()["mse"] = mse

# Schema cho dữ liệu đầu vào của dự đoán điểm thi
class PredictionInput(BaseModel):
  MonHoc: str
  Exam1: float
  Exam2: float
  Exam3: float
  model_type: Literal["cnn", "dnn", "rnn", "lstm"]

# API: Dự đoán điểm thi
@app.post("/predict/exam2")
def predict_score(data: PredictionInput):
  # Chuẩn bị đầu vào
  inputs = np.array([[data.Exam1, data.Exam2, data.Exam3]])
  model_type = data.model_type

  # Lấy model từ danh sách
  model = models.get(model_type)
  if not model:
      return {"error": f"Model '{model_type}' không tồn tại"}
  
  # Dự đoán kết quả
  prediction = model.predict(inputs)
  predicted_score = float(prediction[0][0]) / 10.0  # Chuyển về kiểu float để tránh lỗi

  return {
    "MonHoc": data.MonHoc,
    "Exam1": data.Exam1,
    "Exam2": data.Exam2,
    "Exam3": data.Exam3,
    "Predicted Exam4": predicted_score,
    "Model Used": model_type
  }
model = tf.keras.models.load_model("model-b3/cnn_7_layers.h5")

# Predefined lists for encoding
possible_directions = ['None', 'Đông', 'Tây', 'Nam', 'Bắc', 'Đông - Bắc', 'Đông - Nam', 'Tây - Bắc', 'Tây - Nam']
possible_legals = ['None', 'Hợp đồng mua bán', 'Sổ đỏ', 'Sổ đỏ/ Sổ hồng']

# Initialize encoders
direction_encoder = LabelEncoder()
direction_encoder.fit(possible_directions)
legal_encoder = LabelEncoder()
legal_encoder.fit(possible_legals)

# Initialize scaler
numerical_cols = ['area', 'frontage', 'lat', 'long', 'bedroom', 'toiletCount']
scaler = MinMaxScaler()
# You would typically load the fitted scaler from a saved file in a real-world scenario

# Input validation model
class HousePredictionInput(BaseModel):
    area: float
    bedroom: int
    direction: str
    frontage: float
    lat: float
    long: float
    legal: str
    toiletCount: int

# Prediction endpoint
@app.post("/predict/house-price")
def predict_house_price(data: HousePredictionInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Encode categorical variables
        input_df['direction'] = direction_encoder.transform(input_df['direction'])
        input_df['legal'] = legal_encoder.transform(input_df['legal'])
        
        # Scale numerical features
        input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
        
        # Prepare input for CNN model (add channel dimension)
        X_input = np.expand_dims(input_df.values, axis=-1)
        
        # Predict house price
        predicted_price = model.predict(X_input)
        
        return {
            "input": data.dict(),
            "predicted_price": float(predicted_price[0][0]),
            "price_unit": "tỷ đồng"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
# Load Tokenizer
with open('model-b5/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load mô hình sentiment CNN
sentiment_model = load_model("model-b5/cnn.h5")

class SentimentInput(BaseModel):
    comments: list[str]  # Danh sách bình luận cần phân tích

@app.post("/predict/analyze-sentiment")
def analyze_sentiment(data: SentimentInput):
    try:
        # Tokenize và padding dữ liệu đầu vào
        sequences = tokenizer.texts_to_sequences(data.comments)
        data_padded = pad_sequences(sequences, maxlen=100)  # Giả sử max_len = 100

        # Dự đoán sentiment
        predictions = sentiment_model.predict(data_padded)
        labels = (predictions > 0.5).astype(int)  # 1: Positive, 0: Negative

        # Chuẩn bị kết quả
        results = []
        for i, comment in enumerate(data.comments):
            sentiment = "Positive" if labels[i] == 1 else "Negative"
            confidence = predictions[i][0] * 100
            results.append({
                "comment": comment,
                "sentiment": sentiment,
                "confidence": f"{confidence:.2f}%"
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Chạy server với lệnh: uvicorn main:app --reload
