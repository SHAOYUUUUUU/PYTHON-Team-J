import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from PIL import Image


# ---------- 載入模型與特徵 ----------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("top10_features.pkl", "rb") as f:
    top10_features = pickle.load(f)
with open("optimal_threshold.pkl", "rb") as f:
    optimal_threshold = pickle.load(f)
with open("all_features.pkl", "rb") as f:
    all_features = pickle.load(f)

st.set_page_config(page_title="癌症死亡風險預測", layout="centered")
header_image = Image.open("header_image.png")
st.title("🧬 癌症死亡風險預測系統")
st.markdown("請輸入病患的臨床資訊，我們將預測您未來死亡，請謹慎使用。")

# ---------- 使用者輸入 ----------
with st.form("prediction_form"):
    st.subheader("請輸入以下臨床資訊：")
    user_input = {}
    user_input['CancerStage'] = st.number_input("癌症期數（1~4）", min_value=1, max_value=4, value=2) - 1
    user_input['TumorSize'] = st.number_input("腫瘤大小（單位：cm）", value=3.0)
    user_input['Metastasis'] = st.radio("是否轉移？", options=[0, 1], format_func=lambda x: "否" if x == 0 else "是")
    user_input['Age'] = st.number_input("年齡", min_value=1, max_value=120, value=55)
    user_input['FollowUpMonths'] = st.number_input("追蹤月數", value=24)
    user_input['DaysToSurgery'] = st.number_input("從診斷到手術的天數", value=30)
    user_input['ChemotherapySessions'] = st.number_input("化療次數", value=5)
    user_input['SmokingStatus'] = st.radio("是否吸菸？", options=[0, 1, 2], format_func=lambda x: ["否", "曾經", "是"][x])
    user_input['RadiationSessions'] = st.number_input("放療次數", value=4)
    user_input['AlcoholUse'] = st.radio("是否喝酒？", options=[0, 1, 2], format_func=lambda x: ["否", "偶爾", "時常"][x])

    submitted = st.form_submit_button("🔍 預測風險")

if submitted:
    # ---------- 建立完整特徵向量 ----------
    X_full = pd.DataFrame([0] * len(all_features), index=all_features).T
    for feature in top10_features:
        X_full.at[0, feature] = user_input.get(feature, 0)

    X_new = X_full.copy()
    y_proba = model.predict_proba(X_new)[:, 1][0]
    y_pred = int(y_proba > optimal_threshold)

    # ---------- SHAP 解釋 ----------
    explainer = shap.Explainer(model)
    shap_values = explainer(X_new)

    shap_df = pd.DataFrame({
        '特徵': all_features,
        '數值': X_new.values[0],
        '影響力': shap_values.values[0]
    }).sort_values('影響力', key=abs, ascending=False)

    top_3_factors = shap_df.head(3).to_dict('records')

    # ---------- 風險分級 ----------
    if y_proba <= 0.2:
        risk_level = "低風險"
        risk_emoji = "🟩"
    elif y_proba <= 0.5:
        risk_level = "中風險"
        risk_emoji = "🟨"
    else:
        risk_level = "高風險"
        risk_emoji = "🟥"

    # ---------- 結果展示 ----------
    st.subheader("📊 預測結果")
    st.write(f"死亡機率：**{y_proba*100:.1f}%**")
    st.write(f"預測結果：{risk_emoji} **{'死亡' if y_pred else '存活'}** (閾值={optimal_threshold:.2f})")

    st.subheader("💡 主要風險因素")
    for factor in top_3_factors:
        direction = "↑ 增加風險" if factor['影響力'] > 0 else "↓ 降低風險"
        st.write(f"- {factor['特徵']}: {factor['數值']} ({direction})")

    clinical_advice = {
        '低風險': "✅ 常規隨訪（每6個月一次）",
        '中風險': "⚠️ 建議加強隨訪（每2個月一次）",
        '高風險': "🚨 立即住院治療並啟動多學科會診"
    }
    st.subheader("🏥 臨床建議")
    st.write(clinical_advice[risk_level])
