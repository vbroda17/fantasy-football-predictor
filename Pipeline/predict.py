import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# File paths
DATA_DIR = 'nfl_data'
PREDICTIONS_DIR = 'predictions'
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

PLAYER_INFO_FILE = os.path.join(DATA_DIR, 'player_info.csv')
SCHEDULES_FILE = os.path.join(DATA_DIR, 'nfl_schedules.csv')  # Updated filename
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.csv')
PLAYER_EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'player_embeddings.pkl')
TEAM_EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'team_embeddings.pkl')
MODEL_FILE = os.path.join(DATA_DIR, 'fantasy_score_model.pkl')

def predict_fantasy_score(player_id, opponent_team, player_embeddings, team_embeddings, model):
    # Retrieve embeddings
    player_embedding = player_embeddings.get(player_id)
    team_embedding = team_embeddings.get(opponent_team)
    if player_embedding is None or team_embedding is None:
        # Return None if embeddings are missing
        return None
    # Combine embeddings
    combined_embedding = np.concatenate([player_embedding, team_embedding]).reshape(1, -1)
    # Predict
    predicted_score = model.predict(combined_embedding)
    return predicted_score[0]

def predict_weekly_scores(week_number, season_year, player_info, schedules, player_embeddings, team_embeddings, model):
    # Filter schedules for the specified week and season
    week_schedule = schedules[
        (schedules['season'] == season_year) &
        (schedules['week'] == week_number)
    ]

    if week_schedule.empty:
        # print(f"No games scheduled for week {week_number} in {season_year}.")
        return pd.DataFrame()  # Return empty DataFrame

    # Create a mapping from team to opponent team
    team_opponent_map = {}
    for _, game in week_schedule.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        team_opponent_map[home_team] = away_team
        team_opponent_map[away_team] = home_team

    # List to store predictions
    predictions = []

    # Iterate over all players
    for idx, player in player_info.iterrows():
        player_id = player['player_id']
        player_name = player['player_name']
        team = player['team']
        position = player['position']

        # Get opponent team
        opponent_team = team_opponent_map.get(team)
        if opponent_team is None:
            # No game scheduled for this team in this week
            continue

        # Predict the fantasy score
        predicted_score = predict_fantasy_score(
            player_id=player_id,
            opponent_team=opponent_team,
            player_embeddings=player_embeddings,
            team_embeddings=team_embeddings,
            model=model
        )

        if predicted_score is not None:
            predictions.append({
                'player_id': player_id,
                'player_name': player_name,
                'team': team,
                'position': position,
                'opponent_team': opponent_team,
                'predicted_ppr': predicted_score,
                'season': season_year,
                'week': week_number
            })

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame(predictions)

    return predictions_df

def output_top_players(predictions_df, top_n=20, top_x=5):
    if predictions_df.empty:
        print("No predictions to display.")
        return

    # Sort the predictions by predicted PPR score in descending order
    predictions_df.sort_values(by='predicted_ppr', ascending=False, inplace=True)

    # Output the top N players overall
    print(f"Top {top_n} Players:")
    print(predictions_df.head(top_n)[['player_name', 'team', 'position', 'opponent_team', 'predicted_ppr']])

    print("\nTop Players by Position:")
    # Group by position and output the top X players for each position
    positions = predictions_df['position'].unique()
    for pos in positions:
        pos_df = predictions_df[predictions_df['position'] == pos]
        pos_df = pos_df.head(top_x)
        print(f"\nPosition: {pos}")
        print(pos_df[['player_name', 'team', 'opponent_team', 'predicted_ppr']])

def run_test(player_info, schedules, player_embeddings, team_embeddings, model, processed_data):
    # Only test on the 2024 season
    season_year = 2024

    # Determine the weeks with actual data in processed_data for 2024
    available_weeks = processed_data[
        (processed_data['season'] == season_year) &
        (processed_data['fantasy_points_ppr'].notna())
    ]['week'].unique()

    if len(available_weeks) == 0:
        print(f"No available weeks with actual data for season {season_year}. Exiting.")
        return

    max_week = available_weeks.max()
    print(f"Testing on Season {season_year}, Weeks 1 to {max_week}")

    overall_predictions = pd.DataFrame()

    for week_number in range(1, int(max_week) + 1):
        if week_number not in available_weeks:
            # print(f"No data for week {week_number} in {season_year}. Skipping.")
            continue

        # print(f"Testing Season {season_year}, Week {week_number}")

        # Filter player info for players who played in this week
        week_player_ids = processed_data[
            (processed_data['season'] == season_year) &
            (processed_data['week'] == week_number)
        ]['player_id'].unique()

        if len(week_player_ids) == 0:
            # print(f"No player data for week {week_number} in {season_year}. Skipping.")
            continue

        week_player_info = player_info[player_info['player_id'].isin(week_player_ids)]

        # Predict weekly scores
        predictions_df = predict_weekly_scores(
            week_number=week_number,
            season_year=season_year,
            player_info=week_player_info,
            schedules=schedules,
            player_embeddings=player_embeddings,
            team_embeddings=team_embeddings,
            model=model
        )

        if predictions_df.empty:
            # print(f"No predictions made for week {week_number} in {season_year}. Skipping.")
            continue

        # Merge with actual scores using 'fantasy_points_ppr'
        actual_scores = processed_data[
            (processed_data['season'] == season_year) &
            (processed_data['week'] == week_number)
        ][['player_id', 'fantasy_points_ppr']]

        # Ensure 'player_id' is in both dataframes
        if 'player_id' not in predictions_df.columns or 'player_id' not in actual_scores.columns:
            # print(f"'player_id' not found in predictions or actual scores for week {week_number}. Skipping.")
            continue

        predictions_df = predictions_df.merge(actual_scores, on='player_id', how='left')
        predictions_df.rename(columns={'fantasy_points_ppr': 'actual_ppr'}, inplace=True)

        # Append to overall predictions
        overall_predictions = pd.concat([overall_predictions, predictions_df], ignore_index=True)

    if overall_predictions.empty:
        print("No predictions were made during the test. Exiting.")
        return

    # Calculate overall error metrics
    y_true = overall_predictions['actual_ppr'].values
    y_pred = overall_predictions['predicted_ppr'].values

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    me = np.mean(errors)
    std_error = np.std(errors)

    print(f"\nOverall Test MSE: {mse:.2f}")
    print(f"Overall Test MAE: {mae:.2f}")
    print(f"Overall Mean Error: {me:.2f}")
    print(f"Overall Standard Deviation of Errors: {std_error:.2f}")

    # Add errors to the DataFrame
    overall_predictions['error'] = errors
    overall_predictions['abs_error'] = abs_errors

    # Best and worst overall predictions
    best_prediction = overall_predictions.loc[overall_predictions['abs_error'].idxmin()]
    worst_prediction = overall_predictions.loc[overall_predictions['abs_error'].idxmax()]

    print("\nBest Overall Prediction:")
    print(best_prediction[['player_name', 'team', 'position', 'predicted_ppr', 'actual_ppr', 'error']])

    print("\nWorst Overall Prediction:")
    print(worst_prediction[['player_name', 'team', 'position', 'predicted_ppr', 'actual_ppr', 'error']])

    # Group by position and calculate statistics
    positions = overall_predictions['position'].unique()

    for pos in positions:
        pos_data = overall_predictions[overall_predictions['position'] == pos].copy()
        y_true_pos = pos_data['actual_ppr'].values
        y_pred_pos = pos_data['predicted_ppr'].values
        errors_pos = y_pred_pos - y_true_pos
        abs_errors_pos = np.abs(errors_pos)

        mse_pos = mean_squared_error(y_true_pos, y_pred_pos)
        mae_pos = mean_absolute_error(y_true_pos, y_pred_pos)
        me_pos = np.mean(errors_pos)
        std_error_pos = np.std(errors_pos)

        print(f"\nPosition: {pos}")
        print(f" - MSE: {mse_pos:.2f}")
        print(f" - MAE: {mae_pos:.2f}")
        print(f" - Mean Error: {me_pos:.2f}")
        print(f" - Standard Deviation of Errors: {std_error_pos:.2f}")

        # Best and worst predictions for this position
        pos_data['error'] = errors_pos
        pos_data['abs_error'] = abs_errors_pos

        best_pred_pos = pos_data.loc[pos_data['abs_error'].idxmin()]
        worst_pred_pos = pos_data.loc[pos_data['abs_error'].idxmax()]

        print(f" - Best Prediction:")
        print(best_pred_pos[['player_name', 'team', 'predicted_ppr', 'actual_ppr', 'error']])

        print(f" - Worst Prediction:")
        print(worst_pred_pos[['player_name', 'team', 'predicted_ppr', 'actual_ppr', 'error']])

    # Save overall predictions to CSV
    overall_predictions.to_csv(os.path.join(PREDICTIONS_DIR, 'test_predictions.csv'), index=False)
    # print(f"Test predictions saved to {os.path.join(PREDICTIONS_DIR, 'test_predictions.csv')}")

def add_head_coaches_to_predictions(predictions_df, head_coaches_file):
    """
    Add head coach information to the predictions DataFrame.

    Parameters:
        predictions_df (pd.DataFrame): The DataFrame containing predictions.
        head_coaches_file (str): Path to the CSV file with team and head coach mapping.

    Returns:
        pd.DataFrame: Updated DataFrame with head coach information.
    """
    print("Adding head coaches to predictions...")
    head_coaches_df = pd.read_csv(head_coaches_file)
    head_coaches_dict = dict(zip(head_coaches_df['team'], head_coaches_df['head_coach']))
    predictions_df['head_coach'] = predictions_df['team'].map(head_coaches_dict)
    print("Head coaches added successfully.")
    return predictions_df


def main():
    # Load active players
    player_info = pd.read_csv(PLAYER_INFO_FILE)

    # Load schedules
    schedules = pd.read_csv(SCHEDULES_FILE)

    # Load embeddings
    with open(PLAYER_EMBEDDINGS_FILE, 'rb') as f:
        player_embeddings = pickle.load(f)
    with open(TEAM_EMBEDDINGS_FILE, 'rb') as f:
        team_embeddings = pickle.load(f)

    # Load head coaches
    HEAD_COACHES_FILE = os.path.join(DATA_DIR, 'team_head_coaches.csv')
    head_coaches_df = pd.read_csv(HEAD_COACHES_FILE)
    head_coaches_dict = dict(zip(head_coaches_df['team'], head_coaches_df['head_coach']))

    # Load the trained model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    # Load processed data for actual scores
    processed_data = pd.read_csv(PROCESSED_DATA_FILE)

    # Ensure correct data types
    processed_data['season'] = processed_data['season'].astype(int)
    processed_data['week'] = processed_data['week'].astype(int)

    # Specify the week number or 'TEST'
    week_input = input("Enter the week number for which you want to predict scores (or type 'TEST' to evaluate the model): ")

    if week_input.strip().upper() == 'TEST':
        # Run the testing routine
        run_test(player_info, schedules, player_embeddings, team_embeddings, model, processed_data)
    else:
        # Proceed with predicting for a specific week
        try:
            week_number = int(week_input)
            season_year = 2024  # Update as needed

            # Predict weekly scores
            predictions_df = predict_weekly_scores(
                week_number=week_number,
                season_year=season_year,
                player_info=player_info,
                schedules=schedules,
                player_embeddings=player_embeddings,
                team_embeddings=team_embeddings,
                model=model
            )


            if predictions_df.empty:
                print(f"No predictions available for week {week_number} in {season_year}.")
            else:
                # Output the top players
                output_top_players(predictions_df, top_n=20, top_x=5)

                # Save predictions to the 'predictions' folder
                predictions_file = os.path.join(PREDICTIONS_DIR, f'predictions_week_{week_number}_{season_year}.csv')
                predictions_df.to_csv(predictions_file, index=False)
                print(f'Predictions saved to {predictions_file}')

        except ValueError:
            print("Invalid input. Please enter a valid week number or 'TEST'.")


if __name__ == "__main__":
    main()
