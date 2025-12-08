# model/train_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset: columns = work_stress, emotional_state, social_life, sleep_hours
data = {
    'work_stress': [9, 7, 5, 3, 6, 2, 8, 4, 1, 10],
    'emotional_state': [8, 6, 4, 2, 7, 3, 9, 5, 2, 10],
    'social_life': [3, 5, 7, 9, 4, 8, 2, 6, 10, 1],
    'sleep_hours': [5, 6, 8, 9, 6, 7, 4, 7, 10, 3],
    'stress_level': [9, 7, 5, 3, 6, 4, 8, 5, 2, 10]
}

df = pd.DataFrame(data)

X = df[['work_stress', 'emotional_state', 'social_life', 'sleep_hours']]
y = df['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('stress_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'stress_model.pkl'")
