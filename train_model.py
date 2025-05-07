import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict
import pickle
import os

# Load data
df = pd.read_csv('aurum_recommendation_data.csv')

# Encode categorical features
categorical_cols = ["Skill_Required", "Job_Location", "Subcontractor_Name", "Preference_Match"]
label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Convert outcome
df_encoded["Job_Outcome"] = df["Job_Outcome"].map({"Success": 1, "Failed": -1})

# Drop unused
df_encoded.drop(columns=["Job_ID", "Subcontractor_Skills"], inplace=True)

# Normalize
scaler = MinMaxScaler()
numeric_cols = ["Distance_km", "Job_Duration_hrs", "Experience_Level_Yrs"]
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# Q-learning training
states = df_encoded.drop(columns=["Job_Outcome", "Subcontractor_Name"]).values
rewards = df_encoded["Job_Outcome"].values
actions = df_encoded["Subcontractor_Name"].unique()
q_table = defaultdict(lambda: np.zeros(len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000

for _ in range(episodes):
    for i in range(len(states)):
        state = tuple(states[i])
        reward = rewards[i]
        action_idx = np.random.randint(0, len(actions)) if np.random.rand() < epsilon else np.argmax(q_table[state])
        q_table[state][action_idx] += alpha * (reward + gamma * np.max(q_table[state]) - q_table[state][action_idx])

# Save model
os.makedirs("models", exist_ok=True)
with open("models/q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/actions.pkl", "wb") as f:
    pickle.dump(actions, f)

print("✅ Model and encoders saved in /models")
