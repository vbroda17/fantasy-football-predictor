import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pickle

# File paths
INPUT_FILE = "nfl_data/weekly_game_data_with_ids.csv"
OUTPUT_FILE = "nfl_data/processed_data.csv"

# Embedding file paths
PLAYER_EMBEDDINGS_FILE = "nfl_data/player_embeddings.pkl"
TEAM_EMBEDDINGS_FILE = "nfl_data/team_embeddings.pkl"

def calculate_condensed_stat(row):
    """
    Calculate a condensed stat based on player position using all specified features.
    """
    position = row['position']
    
    if position == 'QB':
        return (
            row['completions'] * 0.1 +
            row['attempts'] * 0.02 +
            row['passing_yards'] * 0.04 +
            row['passing_tds'] * 4 -
            row['interceptions'] * 2 -
            row['sacks'] * 0.5 -
            row['sack_yards'] * 0.05 +
            row['sack_fumbles'] * -0.5 +
            row['sack_fumbles_lost'] * -1 +
            row['passing_air_yards'] * 0.03 +
            row['passing_yards_after_catch'] * 0.02 +
            row['passing_first_downs'] * 0.5 +
            row['passing_epa'] * 1 +
            row['passing_2pt_conversions'] * 2 +
            row['pacr'] * 0.1 +
            row['dakota'] * 0.2 +
            row['carries'] * 0.1 +
            row['rushing_yards'] * 0.1 +
            row['rushing_tds'] * 6 -
            row['rushing_fumbles'] * 2 -
            row['rushing_fumbles_lost'] * 2 +
            row['rushing_first_downs'] * 0.5 +
            row['rushing_epa'] * 1 +
            row['rushing_2pt_conversions'] * 2
        )
    elif position in ['WR', 'TE']:
        return (
            row['receptions'] * 1 +
            row['targets'] * 0.5 +
            row['receiving_yards'] * 0.1 +
            row['receiving_tds'] * 6 -
            row['receiving_fumbles'] * 2 -
            row['receiving_fumbles_lost'] * 2 +
            row['receiving_air_yards'] * 0.03 +
            row['receiving_yards_after_catch'] * 0.02 +
            row['receiving_first_downs'] * 0.5 +
            row['receiving_epa'] * 1 +
            row['receiving_2pt_conversions'] * 2 +
            row['racr'] * 0.1 +
            row['target_share'] * 0.2 +
            row['air_yards_share'] * 0.2 +
            row['wopr'] * 0.3
        )
    elif position == 'RB':
        return (
            row['carries'] * 0.1 +
            row['rushing_yards'] * 0.1 +
            row['rushing_tds'] * 6 -
            row['rushing_fumbles'] * 2 -
            row['rushing_fumbles_lost'] * 2 +
            row['rushing_first_downs'] * 0.5 +
            row['rushing_epa'] * 1 +
            row['rushing_2pt_conversions'] * 2 +
            row['receptions'] * 1 +
            row['targets'] * 0.5 +
            row['receiving_yards'] * 0.1 +
            row['receiving_tds'] * 6 -
            row['receiving_fumbles'] * 2 -
            row['receiving_fumbles_lost'] * 2 +
            row['receiving_air_yards'] * 0.03 +
            row['receiving_yards_after_catch'] * 0.02 +
            row['receiving_first_downs'] * 0.5 +
            row['receiving_epa'] * 1 +
            row['receiving_2pt_conversions'] * 2 +
            row['racr'] * 0.1 +
            row['target_share'] * 0.2 +
            row['air_yards_share'] * 0.2 +
            row['wopr'] * 0.3
        )
    else:
        return 0

EMBEDDING_DIM = 50

def generate_embeddings(data, column, embedding_dim=EMBEDDING_DIM):
    """
    Generate embeddings for a categorical column using PyTorch embeddings.
    Condense embeddings into a single list per row.
    Also returns a mapping from unique values to embeddings.
    """
    print(f"Generating embeddings for {column}...")

    # Map unique values to indices
    unique_values = data[column].dropna().unique()
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}

    # Handle missing values in the column
    data[column] = data[column].fillna("Unknown")
    data[f"{column}_id"] = data[column].map(value_to_index)

    # Create embedding layer
    embedding_layer = nn.Embedding(len(unique_values), embedding_dim)

    # Generate embeddings for each value
    indices = torch.tensor(data[f"{column}_id"].values)
    embeddings = embedding_layer(indices).detach().numpy()

    # Add embeddings as a single list for each row
    data[f"{column}_embedding"] = [list(emb) for emb in embeddings]

    # Create a mapping from value to embedding
    value_embeddings = {}
    for val in unique_values:
        idx = value_to_index[val]
        emb = embedding_layer(torch.tensor(idx)).detach().numpy()
        value_embeddings[val] = emb

    # Drop the ID column
    data = data.drop(columns=[f"{column}_id"])
    return data, value_embeddings

def generate_enriched_team_embeddings(data, team_column, additional_columns=[], embedding_dim=EMBEDDING_DIM):
    print(f"Generating enriched team embeddings for {team_column} with additional columns: {additional_columns}...")

    # Map unique values in the main team column to indices
    unique_teams = data[team_column].dropna().unique()
    team_to_index = {team: idx for idx, team in enumerate(unique_teams)}

    # Handle missing values in the team column
    data[team_column] = data[team_column].fillna("Unknown")
    data[f"{team_column}_id"] = data[team_column].map(team_to_index)

    # Create embedding layer for the main team column
    team_embedding_layer = nn.Embedding(len(unique_teams), embedding_dim)
    team_indices = torch.tensor(data[f"{team_column}_id"].values)
    team_embeddings = team_embedding_layer(team_indices).detach().numpy()

    # Enrich team embeddings with additional columns
    for add_col in additional_columns:
        if add_col not in data.columns:
            continue

        unique_add_values = data[add_col].dropna().unique()
        add_value_to_index = {val: idx for idx, val in enumerate(unique_add_values)}
        data[add_col] = data[add_col].fillna("Unknown")
        data[f"{add_col}_id"] = data[add_col].map(add_value_to_index)

        add_embedding_layer = nn.Embedding(len(unique_add_values), embedding_dim)
        add_indices = torch.tensor(data[f"{add_col}_id"].values)
        add_embeddings = add_embedding_layer(add_indices).detach().numpy()

        team_embeddings = np.concatenate((team_embeddings, add_embeddings), axis=1)

    # Create a dictionary mapping teams to enriched embeddings
    team_embedding_dict = {team: team_embeddings[idx] for team, idx in team_to_index.items()}

    # Drop the ID columns used for embedding
    id_cols = [f"{team_column}_id"] + [f"{col}_id" for col in additional_columns]
    data = data.drop(columns=id_cols)

    return data, team_embedding_dict


def process_data(input_file, output_file):
    """
    Process the data to calculate condensed stats and generate embeddings, then save the processed file.
    """
    print(f"Reading data from {input_file}...")
    data = pd.read_csv(input_file)
    print(f"Data shape: {data.shape}")

    # Fill missing numerical fields with 0
    data.fillna(0, inplace=True)

    # Calculate the condensed stat
    print("Calculating condensed stats...")
    data['condensed_stat'] = data.apply(calculate_condensed_stat, axis=1)

    # Drop all features used in the condensed stat calculations
    features_to_remove = [
        'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks',
        'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
        'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
        'passing_2pt_conversions', 'pacr', 'dakota', 'carries', 'rushing_yards',
        'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_first_downs',
        'rushing_epa', 'rushing_2pt_conversions', 'receptions', 'targets', 'receiving_yards',
        'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
        'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa',
        'receiving_2pt_conversions', 'racr', 'target_share', 'air_yards_share', 'wopr'
    ]
    features_to_remove = [col for col in features_to_remove if col in data.columns]
    data = data.drop(columns=features_to_remove)

    # Initialize dictionaries to hold embeddings
    embeddings_to_save = {}

    # Generate player embeddings
    if 'player_id' in data.columns:
        print("Generating player embeddings...")
        data, player_embeddings = generate_embeddings(data, 'player_id')
        embeddings_to_save['player_id'] = player_embeddings

    # Generate enriched team embeddings (with coaches and referees)
    if 'recent_team' in data.columns:
        print("Generating team embeddings with additional context...")
        # additional_context = ['home_coach', 'away_coach', 'referee_names']
        additional_context = ['home_coach', 'away_coach']
        data, team_embedding_dict = generate_enriched_team_embeddings(
            data, 'recent_team', additional_columns=additional_context
        )
        embeddings_to_save['recent_team'] = team_embedding_dict


    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    data.to_csv(output_file, index=False)
    print("Processing complete.")

    # Save embeddings to files
    print("Saving embeddings...")
    if 'player_id' in embeddings_to_save:
        with open(PLAYER_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings_to_save['player_id'], f)
        print(f"Player embeddings saved to {PLAYER_EMBEDDINGS_FILE}")

    if 'recent_team' in embeddings_to_save:
        with open(TEAM_EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings_to_save['recent_team'], f)
        print(f"Team embeddings saved to {TEAM_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    process_data(INPUT_FILE, OUTPUT_FILE)
