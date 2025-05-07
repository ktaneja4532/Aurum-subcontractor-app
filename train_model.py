import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

# Load dataset
file_path = 'aurum_recommendation_data.csv'
df = pd.read_csv(file_path)

# Encode categorical columns
categorical_cols = ["Skill_Required", "Job_Location", "Subcontractor_Name", "Preference_Match"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode outcome
df["Job_Outcome"] = df["Job_Outcome"].map({"Success": 1, "Failed": -1})

# Drop irrelevant columns
df.drop(columns=["Job_ID", "Subcontractor_Skills"], inplace=True)

# Normalize numeric features
scaler = MinMaxScaler()
numerical_cols = ["Distance_km", "Job_Duration_hrs", "Experience_Level_Yrs"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Prepare data for Q-learning
states = df.drop(columns=["Job_Outcome", "Subcontractor_Name"]).values.astype(float)
rewards = df["Job_Outcome"].values.astype(int)
actions = df["Subcontractor_Name"].unique()
q_table = defaultdict(lambda: np.zeros(len(actions)))

# Q-learning training
alpha, gamma, epsilon = 0.1, 0.9, 0.2
episodes = 1000
for _ in range(episodes):
    for i in range(len(states)):
        state = tuple(states[i])
        reward = rewards[i]
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(0, len(actions))
        else:
            action_idx = np.argmax(q_table[state])

        next_state = state
        q_old = q_table[state][action_idx]
        q_next_max = np.max(q_table[next_state])
        q_table[state][action_idx] = q_old + alpha * (reward + gamma * q_next_max - q_old)

# Save trained components
with open('models/q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Training complete. Models saved to models/ folder.")
