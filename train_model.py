# -*- coding: utf-8 -*-
"""train_model.py

Q-learning model trainer for subcontractor recommendation.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --- Step 1: Load data ---
data_path = "aurum_recommendation_data.csv"
df = pd.read_csv(data_path)

# --- Step 2: Encode categorical columns ---
categorical_cols = ["Skill_Required", "Job_Location", "Subcontractor_Name", "Preference_Match"]
label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# --- Step 3: Normalize numerical columns ---
numerical_cols = ["Distance_km", "Job_Duration_hrs", "Experience_Level_Yrs"]
scaler = MinMaxScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# --- Step 4: Convert Job Outcome to reward ---
df_encoded["Job_Outcome"] = df["Job_Outcome"].map({"Success": 1, "Failed": -1})

# --- Step 5: Initialize Q-learning ---
q_table = {}  # Use plain dict to avoid pickle issues

states = df_encoded[["Skill_Required", "Job_Location", "Distance_km", "Preference_Match",
                     "Job_Duration_hrs", "Experience_Level_Yrs"]].values
actions = df_encoded["Subcontractor_Name"].unique()
rewards = df_encoded["Job_Outcome"].values

action_len = len(actions)

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

# --- Step 6: Train Q-learning model ---
for ep in range(episodes):
    for i in range(len(states)):
        state = tuple(states[i])
        action = df_encoded["Subcontractor_Name"].iloc[i]
        action_idx = np.where(actions == action)[0][0]
        reward = rewards[i]

        if state not in q_table:
            q_table[state] = np.zeros(action_len)

        current_q = q_table[state][action_idx]
        max_future_q = np.max(q_table[state])
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        q_table[state][action_idx] = new_q

# --- Step 7: Save model artifacts ---
os.makedirs("models", exist_ok=True)

with open("models/q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Q-learning training complete. Model files saved in /models.")
