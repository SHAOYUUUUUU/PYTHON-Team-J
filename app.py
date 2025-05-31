import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle

# ====== 載入模型與參數 ======
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("top10_features.pkl", "rb") as f:
    top10_features = pickle.load(f)

with open("optimal_threshold.pkl", "rb") as f:
    optimal_threshold = pickle.load(f)

# ====== 建立使用者輸入介面 ======
st.title("🔍 癌症死亡風險預測")
st.write("請填寫以下欄位以進行預測：")

user_input = {}
user_input['AlcoholUse'] = st.selectbox("飲酒習慣", ["Never", "Occasional", "Heavy"])
user_input['SmokingStatus'] = st.selectbox("抽菸狀況", ["Never", "Former", "Current"])
user_input['CancerStage'] = st.selectbox("癌症期別", ["Stage I", "Stage II", "Stage III", "Stage IV"])
user_input['Metastasis=Yes'] = st.selectbox("是否有轉移", ["Yes", "No"]) == "Yes"
user_input['DaysToSurgery'] = st.number_input("診斷到手術的天數", min_value=0, max_value=500, value=30)
user_input['HasComorbidity'] = st.selectbox("是否有共病", ["是", "否"]) == "是"
user_input['Age'] = st.number_input("年齡", min_value=18, max_value=120, value=60)
user_input['BMI'] = st.number_input("BMI 指數", min_value=10.0, max_value=50.0, value=22.0)
user_input['BloodPressure'] = st.number_input("血壓", min_value=80, max_value=200, value=120)
user_input['Cholesterol'] = st.number_input("膽固醇", min_value=100, max_value=300, value=180)

if st.button("🚀 預測死亡風險"):
    # ====== 轉換為 DataFrame 並處理欄位 ======
    X_user = pd.DataFrame([user_input])

    label_maps = {
        "AlcoholUse": {"Never": 2, "Occasional": 1, "Heavy": 0},
        "SmokingStatus": {"Never": 0, "Former": 1, "Current": 2},
        "CancerStage": {"Stage I": 0, "Stage II": 1, "Stage III": 2, "Stage IV": 3},
    }

    for col, mapping in label_maps.items():
        if col in X_user.columns:
            X_user[col] = X_user[col].map(mapping)

    X_user['Metastasis=Yes'] = X_user['Metastasis=Yes'].astype(int)
    X_user['HasComorbidity'] = X_user['HasComorbidity'].astype(int)

    for col in top10_features:
        if col not in X_user.columns:
            X_user[col] = 0
    X_user = X_user[top10_features]

    # ====== 預測與 SHAP 分析 ======
    y_proba = model.predict_proba(X_user)[:, 1][0]
    y_pred = int(y_proba >= optimal_threshold)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_user)

    shap_df = pd.DataFrame({
        '特徵': top10_features,
        '數值': X_user.values[0],
        '影響力': shap_values.values[0]
    }).sort_values('影響力', key=abs, ascending=False)

    top_3_factors = shap_df.head(3)

    # ====== 顯示結果 ======
    st.subheader("📊 預測結果")
    st.write(f"死亡概率: **{y_proba*100:.1f}%**")
    st.write(f"預測結果: {'🟥 高風險(死亡)' if y_pred else '🟩 低風險(存活)'} (閾值 = {optimal_threshold:.2f})")

    st.subheader("💡 主要風險因素")
    for _, row in top_3_factors.iterrows():
        direction = "↑ 增加" if row['影響力'] > 0 else "↓ 減少"
        st.write(f"- {row['特徵']}: {row['數值']} ({direction}風險)")

    # ====== 臨床建議 ======
    if y_proba <= 0.2:
        advice = "常規隨訪（每6個月一次）"
    elif y_proba <= 0.5:
        advice = "加強隨訪（每2個月一次）"
    else:
        advice = "🔺 建議立即住院並啟動多學科會診"

    st.subheader("🏥 臨床建議")
    st.write(advice)
