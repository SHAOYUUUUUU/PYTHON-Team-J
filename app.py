import streamlit as st
import pandas as pd
import pickle
import shap

# ---------- 載入模型與設定 ----------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("top10_features.pkl", "rb") as f:
    top10_features = pickle.load(f)

with open("optimal_threshold.pkl", "rb") as f:
    optimal_threshold = pickle.load(f)

with open("all_features.pkl", "rb") as f:
    all_features = pickle.load(f)

# ---------- Streamlit UI ----------
st.title("🩺 亮瑜醫院癌症死亡風險預測")
st.write("請填寫以下資訊，系統將預測病患死亡風險並提供臨床建議")

# ---------- 使用者輸入 ----------
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
    y_proba = model.predict_proba(X_new)[0][1]
    y_pred = int(y_proba >= optimal_threshold)

    # 風險等級
    if y_proba <= 0.2:
        risk_level = "低風險"
        risk_level_full = "低風險，安啦安啦"
        risk_emoji = "🟩"
    elif y_proba <= 0.5:
        risk_level = "中風險"
        risk_level_full = "中風險，要小心喔老哥"
        risk_emoji = "🟨"
    else:
        risk_level = "高風險"
        risk_level_full = "高風險，沒救了啦，下輩子好好做人"
        risk_emoji = "🟥"

    # 顯示預測結果
    st.subheader("📊 預測結果")
    st.write(f"死亡機率：**{y_proba*100:.1f}%**")
    st.write(f"預測結果：{risk_emoji} **{'死亡' if y_pred else '存活'}**（閾值={optimal_threshold:.2f}）")

    # SHAP 解釋（前三大風險因素）
    explainer = shap.Explainer(model)
    shap_values = explainer(X_new)

    shap_df = pd.DataFrame({
        '特徵': top10_features,
        '數值': X_new[top10_features].values[0],
        '影響力': shap_values.values[0]
    }).sort_values('影響力', key=abs, ascending=False)

    top_3 = shap_df.head(3).to_dict('records')
    st.subheader("💡 主要風險因素（前3名）")
    for item in top_3:
        direction = "↑ 增加風險" if item['影響力'] > 0 else "↓ 降低風險"
        st.write(f"- `{item['特徵']}`：{item['數值']}（{direction}）")

    # 臨床建議
    advice = {
        '低風險': "✅ 常規隨訪（每6個月一次）",
        '中風險': "⚠️ 建議加強隨訪（每2個月一次）",
        '高風險': "🚨 立即住院治療並啟動多學科會診"
    }

    st.subheader("🏥 臨床建議")
    st.write(f"{risk_level_full}：{advice[risk_level]}")
