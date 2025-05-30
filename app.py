# -*- coding: utf-8 -*-
"""app.py - Streamlit UI for Aurum subcontractor recommendation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cdist
from textblob import TextBlob
import os

# Load trained Q-table
with open("models/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Load label encoders
with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load raw data for dropdowns
df = pd.read_csv("aurum_recommendation_data.csv")

st.title("🔍 Subcontractor Recommendation System")
st.markdown("Enter job details to get the best subcontractor recommendation.")

# --- User Inputs ---
skills = df["Skill_Required"].unique().tolist()
locations = df["Job_Location"].unique().tolist()
preference_opts = ["Yes", "No"]

skill = st.selectbox("Skill Required", skills)
location = st.selectbox("Job Location", locations)
distance = st.slider("Distance (km)", 0, 50, 5)
preference = st.radio("Preference Match", preference_opts)
duration = st.slider("Job Duration (hrs)", 1, 12, 4)
experience = st.slider("Experience Level (yrs)", 0, 10, 2)

# Encode categorical inputs
input_encoded = [
    label_encoders['Skill_Required'].transform([skill])[0],
    label_encoders['Job_Location'].transform([location])[0],
    label_encoders['Preference_Match'].transform([preference])[0]
]

# Normalize numeric inputs
numeric_input = np.array([[distance, duration, experience]])
normalized = scaler.transform(numeric_input)[0]

# Final input state for Q-learning
new_job_state = tuple([
    input_encoded[0],
    input_encoded[1],
    normalized[0],
    input_encoded[2],
    normalized[1],
    normalized[2]
])

# --- Recommend Best Match ---
def get_closest_q_values(input_state, q_table):
    input_state = np.array(input_state).reshape(1, -1)
    q_keys = np.array([list(k) for k in q_table.keys()])
    distances = cdist(input_state, q_keys)
    closest_idx = np.argmin(distances)
    closest_state = tuple(q_keys[closest_idx])
    return q_table[closest_state]

# Fetch actions
actions = df["Subcontractor_Name"].unique()
actions_encoded = label_encoders["Subcontractor_Name"].transform(actions)

if st.button("🔎 Recommend Subcontractor"):
    if new_job_state in q_table:
        q_values = q_table[new_job_state]
    else:
        q_values = get_closest_q_values(new_job_state, q_table)

    # Get top 3 recommendations
    top_indices = np.argsort(q_values)[::-1][:3]
    top_names = label_encoders["Subcontractor_Name"].inverse_transform(top_indices)

    st.success("✅ Top Recommended Subcontractors:")
    for i, idx in enumerate(top_indices):
        st.write(f"{i+1}. {top_names[i]} (Q-Score: {round(q_values[idx], 4)})")

    recommended_idx = top_indices[0]  # For feedback

    # --- Feedback Section ---
    st.subheader("📝 Provide Feedback")
    categories = ["Punctuality", "Communication", "Skill Fit", "Professionalism", "Overall Satisfaction"]
    ratings = [st.slider(cat, 1, 5, 3) for cat in categories]
    feedback_text = st.text_area("💬 Additional Comments")

    if st.button("Submit Feedback"):
        avg_rating = sum(ratings) / len(ratings)
        rating_reward = (avg_rating - 3) / 2
        sentiment_score = TextBlob(feedback_text).sentiment.polarity
        final_reward = (rating_reward + sentiment_score) / 2 if feedback_text.strip() else rating_reward
        final_reward = max(min(final_reward, 1), -1)

        current_q = q_values[recommended_idx]
        max_future_q = np.max(q_values)
        q_values[recommended_idx] = current_q + 0.1 * (final_reward + 0.9 * max_future_q - current_q)

        st.success(f"🧠 Feedback reward computed: {round(final_reward, 3)}")

        if final_reward < -0.2:
            st.warning("⚠️ Negative experience detected. Suggesting alternatives:")
            q_copy = q_values.copy()
            q_copy[recommended_idx] = -np.inf
            alt_indices = np.argsort(q_copy)[::-1][:2]
            alt_names = label_encoders["Subcontractor_Name"].inverse_transform(alt_indices)
            for name in alt_names:
                st.write(f"🔁 {name}")
        else:
            st.info("Thank you! Feedback updated the model.")
