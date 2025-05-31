import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ---------- 讀取資料與訓練模型 ----------
df = pd.read_csv("china_cancer_patients_synthetic.csv")

target_col = "SurvivalStatus"
df[target_col] = df[target_col].map({'Alive': 0, 'Deceased': 1})

X = df.drop(columns=[target_col])
y = df[target_col]

# 類別欄位轉換
for col in X.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

model = xgb.XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=3.5,
    random_state=42,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    learning_rate=0.05
)
model.fit(X, y)

# ---------- 參數定義 ----------
top10_features = ['CancerStage', 'TumorSize', 'Metastasis', 'Age', 'FollowUpMonths',
                  'DaysToSurgery', 'ChemotherapySessions', 'SmokingStatus',
                  'RadiationSessions', 'AlcoholUse']
all_features = X.columns.tolist()
optimal_threshold = 0.4782  # ← 根據你訓練結果設定

# ---------- Streamlit UI ----------
st.title("🩺 亮瑜醫院癌症死亡率預測系統")
st.write("請輸入病患的相關資料，系統將預測其死亡風險，並提供臨床建議。沒事不要亂用 ><")

# ---------- 中文輸入介面 ----------
user_input = {}

user_input['CancerStage'] = st.number_input("癌症期數（例：1～4）", min_value=1, max_value=4, step=1) - 1
user_input['TumorSize'] = st.number_input("腫瘤大小（單位：cm）", min_value=0.0, step=0.1)
user_input['Metastasis'] = st.number_input("是否轉移？（是=1，否=0）", min_value=0, max_value=1, step=1)
user_input['Age'] = st.number_input("年齡", min_value=0, step=1)
user_input['FollowUpMonths'] = st.number_input("追蹤月數", min_value=0, step=1)
user_input['DaysToSurgery'] = st.number_input("從診斷到手術的天數", min_value=0, step=1)
user_input['ChemotherapySessions'] = st.number_input("化療次數", min_value=0, step=1)
user_input['SmokingStatus'] = st.number_input("是否吸菸？（是=2，曾經=1，否=0）", min_value=0, max_value=2, step=1)
user_input['RadiationSessions'] = st.number_input("放療次數", min_value=0, step=1)
user_input['AlcoholUse'] = st.number_input("是否喝酒？（時常=2，偶爾=1，否=0）", min_value=0, max_value=2, step=1)

# ---------- 預測 ----------
if st.button("🔍 預測死亡風險"):
    X_new = pd.DataFrame([user_input])[top10_features]
    X_new = X_new.reindex(columns=all_features, fill_value=0)

    y_proba = model.predict_proba(X_new)[:, 1][0]
    y_pred = int(y_proba >= optimal_threshold)

    # 判斷風險等級
    if y_proba <= 0.2:
        risk_level_key = "低風險"
        risk_display = "低風險，安啦安啦"
        risk_emoji = "🟩"
    elif y_proba <= 0.5:
        risk_level_key = "中風險"
        risk_display = "中風險，要小心喔老哥"
        risk_emoji = "🟨"
    else:
        risk_level_key = "高風險"
        risk_display = "高風險，沒救了啦，下輩子好好做人"
        risk_emoji = "🟥"

    # 臨床建議對應表（記得這要在 if 裡）
    clinical_advice = {
        "低風險": "✅ 常規隨訪（每6個月一次）",
        "中風險": "⚠️ 建議加強隨訪（每2個月一次）",
        "高風險": "🚨 立即住院治療並啟動多學科會診"
    }

    st.subheader("📊 預測結果")
    st.write(f"死亡機率：**{y_proba*100:.1f}%**")
    st.write(f"預測結果：{risk_emoji} **{'死亡' if y_pred else '存活'}**")

    st.subheader("🏥 臨床建議")
    st.write(f"{risk_emoji} {risk_display}")
    st.write(clinical_advice[risk_level_key])
