import os
import joblib
import streamlit as st
import numpy as np

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 生成 `best_model.pkl` 的正确路径
model_path = os.path.join(current_dir, "best_model.pkl")

# 确保文件存在后再加载
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"模型文件未找到: {model_path}")

# Streamlit UI
st.title("My first ML App (Study on Imbalanced Data Classification)")

# 输入特征
features = []
for i in range(11):  # 依据你的数据集调整
    value = st.number_input(f"Feature_{i}", value=0.0)
    features.append(value)

# 预测
if st.button("Predict"):
    if os.path.exists(model_path):
        prediction = model.predict([np.array(features)])
        st.write(f"Predicted Class: {prediction[0]}")
    else:
        st.error("无法进行预测，模型文件不存在。")
