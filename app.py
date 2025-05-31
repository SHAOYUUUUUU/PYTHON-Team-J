import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# 載入模型
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("optimal_threshold.pkl", "rb") as f:
    optimal_threshold = pickle.load(f)

# 直接定義 top10 特徵
# 依照你的設計，用戶只會輸入這 10 個特徵

TOP10_FEATURES = [
    'CancerStage', 'TumorSize', 'Metastasis', 'Age', 'FollowUpMonths',
    'DaysToSurgery', 'ChemotherapySessions', 'SmokingStatus', 'RadiationSessions', 'AlcoholUse'
]

st.set_page_config(page_title="癌症預測模型 App")
st.title("癌症死亡預測 + 臨牀建議")

# 用戶輸入區塊
st.header("請輸入用戶健康資料")

user_input = {}
user_input['CancerStage'] = st.slider("癌症期數", 1, 4, 2) - 1
user_input['TumorSize'] = st.number_input("腫瘤大小 (cm)", min_value=0.0)
user_input['Metastasis'] = st.selectbox("是否轉移", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
user_input['Age'] = st.number_input("年齡", min_value=0)
user_input['FollowUpMonths'] = st.number_input("追蹤月數", min_value=0)
user_input['DaysToSurgery'] = st.number_input("診斷到手術的天數", min_value=0)
user_input['ChemotherapySessions'] = st.number_input("化療次數", min_value=0)
user_input['SmokingStatus'] = st.selectbox("吸菸情況", [0, 1, 2], format_func=lambda x: ["否", "曾經", "是"][x])
user_input['RadiationSessions'] = st.number_input("放療次數", min_value=0)
user_input['AlcoholUse'] = st.selectbox("飲酒情況", [0, 1, 2], format_func=lambda x: ["否", "偶爾", "時常"][x])

# 正式預測結果
if st.button("執行預測"):
    X_new = pd.DataFrame([user_input])[TOP10_FEATURES]
    y_proba = model.predict_proba(X_new)[:, 1]
    y_pred = int(y_proba[0] >= optimal_threshold)

    st.subheader("預測結果")
    st.write(f"死亡概率: {y_proba[0]:.2f}")
    st.write("預測結論：", "😓 高風險 (預測為死亡)" if y_pred == 1 else "😊 低風險 (預測為存活)")

    st.subheader("臨牀建議")
    if y_pred == 1:
        st.markdown("""
        請評估患者化療、放療不良反應，增加追蹤風險與命題討論
        """)
    else:
        st.markdown("""
        繼續追蹤患者健康狀況，檢視腹空、水血指標以及學習專科資訊
        """)
