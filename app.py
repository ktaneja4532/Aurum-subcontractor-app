# -*- coding: utf-8 -*-
"""app.py: Aurum Subcontractor Recommendation System

This app recommends the best subcontractors based on job input,
and updates its intelligence based on user feedback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cdist
from textblob import TextBlob
from collections import defaultdict

# -------------------------------
# Load Saved Models and Data
# -------------------------------
with open("models/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load reference data
df = pd.read_csv("aurum_recommendation_data.csv")
actions = df["Subcontractor_Name"].unique()
actions_encoded = label_encoders["Subcontractor_Name"].transform(actions)

# -------------------------------
# Helper: Recommend Top N
# -------------------------------
def recommend_top_n_matches(input_state, q_table, actions_encoded, n=3):
    input_state = np.array(input_state).reshape(1, -1)
    q_keys = np.array([list(k) for k in q_table.keys()])
    distances = cdist(input_state, q_keys, metric='euclidean')
    closest_idx = np.argmin(distances)
    closest_state = tuple(q_keys[closest_idx])
    q_values = q_table[closest_state]
    top_n_indices = np.argsort(q_values)[::-1][:n]
    top_names = label_encoders["Subcontractor_Name"].inverse_transform(top_n_indices)
    return [(name, q_values[i]) for name, i in zip(top_names, top_n_indices)]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔧 Aurum Subcontractor Recommendation")
st.markdown("Provide job details to get AI-based subcontractor recommendations.")

col1, col2 = st.columns(2)
with col1:
    skill = st.selectbox("Skill Required", df["Skill_Required"].unique())
    location = st.selectbox("Job Location", df["Job_Location"].unique())
    preference = st.selectbox("Preference Match", ["Yes", "No"])

with col2:
    distance = st.slider("Distance (km)", 0, 50, 5)
    duration = st.slider("Job Duration (hrs)", 1, 12, 4)
    experience = st.slider("Experience Required (yrs)", 0, 20, 3)

# Encode + Normalize inputs
input_encoded = [
    label_encoders['Skill_Required'].transform([skill])[0],
    label_encoders['Job_Location'].transform([location])[0],
    label_encoders['Preference_Match'].transform([preference])[0]
]
numeric = np.array([[distance, duration, experience]])
scaled = scaler.transform(numeric)[0]

new_job_state = tuple([
    input_encoded[0], input_encoded[1], scaled[0],
    input_encoded[2], scaled[1], scaled[2]
])

if st.button("🔍 Recommend Subcontractors"):
    top3 = recommend_top_n_matches(new_job_state, q_table, actions_encoded)
    st.subheader("✅ Top 3 Subcontractor Matches")
    for name, score in top3:
        st.write(f"- **{name}** (Q-Score: {round(score, 3)})")

    st.session_state["selected_state"] = new_job_state
    st.session_state["selected_idx"] = label_encoders['Subcontractor_Name'].transform([top3[0][0]])[0]

# -------------------------------
# Feedback Section
# -------------------------------
st.markdown("---")
st.subheader("📝 Rate the selected subcontractor")
categories = ["Punctuality", "Communication", "Skill Fit", "Professionalism", "Overall Satisfaction"]
ratings = [st.slider(f"{cat} Rating", 1, 5, 3) for cat in categories]
feedback_text = st.text_area("💬 Additional Comments (optional)")

if st.button("Submit Feedback"):
    avg_rating = sum(ratings) / len(ratings)
    rating_reward = (avg_rating - 3) / 2
    sentiment_score = TextBlob(feedback_text).sentiment.polarity if feedback_text.strip() else 0
    final_reward = np.clip((rating_reward + sentiment_score) / 2 if feedback_text else rating_reward, -1, 1)

    state = st.session_state.get("selected_state")
    idx = st.session_state.get("selected_idx")

    if state and idx is not None:
        current_q = q_table[state][idx]
        max_future_q = np.max(q_table[state])
        alpha, gamma = 0.1, 0.9
        q_table[state][idx] = current_q + alpha * (final_reward + gamma * max_future_q - current_q)

        st.success(f"🧠 Feedback reward computed: {round(final_reward, 3)}")

        if final_reward < -0.2:
            q_copy = q_table[state].copy()
            q_copy[idx] = -np.inf
            alt_indices = np.argsort(q_copy)[::-1][:2]
            alt_names = label_encoders["Subcontractor_Name"].inverse_transform(alt_indices)
            st.warning("⚠️ Negative experience detected. Suggested Alternatives:")
            for i, alt in enumerate(alt_names):
                st.write(f"- {alt} (Q-Score: {round(q_copy[alt_indices[i]], 3)})")
        else:
            st.success("✅ Thank you! Feedback recorded and model updated.")
    else:
        st.warning("⚠️ Please request a recommendation first.")
