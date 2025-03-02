# train_model.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# File paths
DATA_DIR = 'nfl_data'

PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.csv')
PLAYER_EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'player_embeddings.pkl')
TEAM_EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'team_embeddings.pkl')
MODEL_FILE = os.path.join(DATA_DIR, 'fantasy_score_model.pkl')

def load_data():
    # Load processed data
    data = pd.read_csv(PROCESSED_DATA_FILE)

    # Load embeddings
    with open(PLAYER_EMBEDDINGS_FILE, 'rb') as f:
        player_embeddings = pickle.load(f)
    with open(TEAM_EMBEDDINGS_FILE, 'rb') as f:
        team_embeddings = pickle.load(f)

    return data, player_embeddings, team_embeddings

def prepare_features(data, player_embeddings, team_embeddings):
    X = []
    y = []

    for idx, row in data.iterrows():
        player_id = row['player_id']
        opponent_team = row['opponent_team']
        fantasy_points_ppr = row['fantasy_points_ppr']

        player_embedding = player_embeddings.get(player_id)
        team_embedding = team_embeddings.get(opponent_team)

        if player_embedding is not None and team_embedding is not None:
            combined_embedding = np.concatenate([player_embedding, team_embedding])
            X.append(combined_embedding)
            y.append(fantasy_points_ppr)
        else:
            # Skip if embeddings are missing
            continue

    X = np.array(X)
    y = np.array(y)
    return X, y

def train_model(X, y):
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training the model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MSE: {mse:.2f}")
    print(f"Validation MAE: {mae:.2f}")

    return model

def save_model(model):
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_FILE}")

def main():
    data, player_embeddings, team_embeddings = load_data()
    X, y = prepare_features(data, player_embeddings, team_embeddings)
    model = train_model(X, y)
    save_model(model)

if __name__ == "__main__":
    main()
