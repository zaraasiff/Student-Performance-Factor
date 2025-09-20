import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------
# Fake simple model (replace with trained one if you want)
# ----------------------------
model = LinearRegression()
# Train a dummy model for demo (Hours_Studied vs Score)
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([40,45,50,55,60,65,70,75,80,85])
model.fit(X, y)

# ----------------------------
# Custom CSS for background + styling
# ----------------------------
page_bg = """
<style>
body {
    background-color: #0f172a;
    color: white;
    font-family: 'Trebuchet MS', sans-serif;
}
.stButton>button {
    background: linear-gradient(to right, #4f46e5, #9333ea);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-size: 16px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("📊 Student Performance Predictor")
st.markdown("### Enter your study details to check performance 🚀")

# ----------------------------
# Inputs
# ----------------------------
hours = st.slider("⏳ Hours Studied per Week", 0, 20, 5)
attendance = st.slider("📅 Attendance (%)", 50, 100, 80)
previous_score = st.slider("📖 Previous Exam Score", 0, 100, 60)

# ----------------------------
# Prediction
# ----------------------------
pred_score = model.predict([[hours]])[0]

# Add effect of attendance + previous score
pred_score = pred_score * (attendance/100) + (previous_score * 0.3)
pred_score = round(pred_score, 1)

st.metric("🎯 Predicted Exam Score", f"{pred_score}/100")

# ----------------------------
# Performance Message
# ----------------------------
if pred_score >= 75:
    st.success("🔥 Excellent Work! Keep it up.")
elif pred_score >= 50:
    st.warning("⚠ Average Performance. You can do better!")
else:
    st.error("❌ Below Average - Needs Serious Improvement.")

# ----------------------------
# Recommendations
# ----------------------------
st.subheader("💡 Recommendations")
if pred_score >= 75:
    st.write("- Maintain consistency in study hours")
    st.write("- Revise topics weekly for sustained performance")
elif pred_score >= 50:
    st.write("- Increase study hours gradually")
    st.write("- Focus on weak areas with extra practice")
else:
    st.write("- Dedicate at least 10+ hours per week to studies")
    st.write("- Seek help from teachers/mentors")
    st.write("- Improve attendance and class participation")
