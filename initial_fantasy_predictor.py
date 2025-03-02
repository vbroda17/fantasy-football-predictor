import nfl_data_py as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Fetch Historical Data
seasons = [2022, 2023, 2024]  # Use multiple seasons for training
df = nfl.import_weekly_data(seasons)

# Step 2: Feature Engineering
# Aggregate stats from past games to predict future performance
df['total_yards'] = df['passing_yards'] + df['rushing_yards'] + df['receiving_yards']
df['total_touchdowns'] = df['passing_tds'] + df['rushing_tds'] + df['receiving_tds']

# Calculate fantasy points
df['fantasy_points'] = (
    df['passing_yards'] * 0.04 +
    df['passing_tds'] * 4 +
    df['interceptions'] * -1 +
    df['rushing_yards'] * 0.1 +
    df['rushing_tds'] * 6 +
    df['receiving_yards'] * 0.1 +
    df['receiving_tds'] * 6 +
    df['sack_fumbles_lost'] * -2 +
    df['rushing_fumbles_lost'] * -2 +
    df['receiving_fumbles_lost'] * -2
)

# Drop rows with missing data
df = df.dropna(subset=['player_id', 'fantasy_points'])

# Step 3: Create Aggregated Features Based on Past Performance
agg_features = df.groupby('player_id').agg({
    'fantasy_points': ['mean', 'std'],  # Average and variability of fantasy points
    'total_yards': ['mean'],            # Average total yards
    'total_touchdowns': ['mean'],       # Average total touchdowns
    'week': 'count'                     # Number of games played
}).reset_index()

agg_features.columns = [
    'player_id', 'avg_fantasy_points', 'std_fantasy_points',
    'avg_total_yards', 'avg_total_touchdowns', 'games_played'
]

# Keep only players with enough historical data
agg_features = agg_features[agg_features['games_played'] > 5]

# Merge aggregated features back to player-level data for model training
df = df.merge(agg_features, on='player_id', how='left')

# Step 4: Prepare Data for Model Training
# Use aggregated features as predictors for the last game's fantasy points
features = [
    'avg_fantasy_points', 'std_fantasy_points', 'avg_total_yards',
    'avg_total_touchdowns', 'games_played'
]
df = df.dropna(subset=features + ['fantasy_points'])

X = df[features]
y = df['fantasy_points']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model Performance
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Step 6: Predict Future Scores
# Use the aggregated stats for players to predict Week 11 of 2024
future_week = 11
future_players = nfl.import_players()  # Fetch player data

# Merge player-level stats for prediction
predictions = agg_features.copy()
predictions['predicted_fantasy_points'] = model.predict(predictions[features])

# Add player names for better readability
if 'gsis_id' in future_players.columns:
    future_players['gsis_id'] = future_players['gsis_id'].astype(str)
    predictions = predictions.merge(
        future_players[['gsis_id', 'display_name']],
        left_on='player_id',
        right_on='gsis_id',
        how='left'
    )

# Step 7: Rank Players by Predicted Fantasy Points
ranked_players = predictions[['display_name', 'predicted_fantasy_points']].sort_values(
    by='predicted_fantasy_points', ascending=False
)

# Display Top 30 Players
print(ranked_players.head(30))

