from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

custom_objects = {
    "mse": MeanSquaredError()
}


# Khởi tạo FastAPI
app = FastAPI()

# Load models
models = {
  "cnn": tf.keras.models.load_model("models/cnn_model.h5", custom_objects=custom_objects),
  "dnn": tf.keras.models.load_model("models/dnn_model.h5", custom_objects=custom_objects),
  "rnn": tf.keras.models.load_model("models/rnn_model.h5", custom_objects=custom_objects),
  "lstm": tf.keras.models.load_model("models/lstm_model.h5", custom_objects=custom_objects),
}


# Schema cho đầu vào
class PredictionInput(BaseModel):
  MonHoc: str
  Exam1: float
  Exam2: float
  Exam3: float
  model_type: Literal["cnn", "dnn", "rnn", "lstm"]

@app.post("/predict")
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
    predicted_score = float(prediction[0][0])  # Chuyển về kiểu float để tránh lỗi

    return {
        "MonHoc": data.MonHoc,
        "Exam1": data.Exam1,
        "Exam2": data.Exam2,
        "Exam3": data.Exam3,
        "Predicted Exam4": predicted_score,
        "Model Used": model_type
    }


# Chạy server với lệnh: uvicorn main:app --reload
