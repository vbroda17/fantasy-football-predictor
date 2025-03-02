import nfl_data_py as nfl
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Directory to save data files
DATA_DIR = "nfl_data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 1. Fetch active fantasy players for the current year and save their data
def fetch_and_save_active_fantasy_players(current_year=2024):
    try:
        players = nfl.__import_rosters(years=[current_year], release='seasonal')
        # Filter for active players and specific fantasy positions
        active_fantasy_players = players[
            (players['status'] == 'ACT') & 
            (players['position'].isin(['QB', 'WR', 'RB', 'TE']))
        ]
        
        # Drop unnecessary columns
        columns_to_drop = [
            'depth_chart_position', 'jersey_number', 'college', 'espn_id', 'sportradar_id',
            'yahoo_id', 'rotowire_id', 'pff_id', 'pfr_id', 'fantasy_data_id', 'sleeper_id',
            'headshot_url', 'ngs_position', 'status_description_abbr', 'football_name',
            'esb_id', 'gsis_it_id', 'smart_id', 'entry_year'
        ]
        active_fantasy_players = active_fantasy_players.drop(columns=columns_to_drop)

        # Save the data to CSV
        active_fantasy_players.to_csv(f"{DATA_DIR}/active_fantasy_players.csv", index=False)
        print(f"Saved active fantasy players data to {DATA_DIR}/active_fantasy_players.csv")
        return active_fantasy_players
    except AttributeError as e:
        print("Error: The nfl_data_py module does not provide a function for rosters.")
        raise e

def fetch_and_save_weekly_game_data_with_ids(year_range, active_players):
    try:
        # Fetch the weekly data for the specified year range
        print(f"Fetching weekly game data for years: {year_range}...")
        weekly_game_data = nfl.import_weekly_data(year_range)
        
        # Filter for only regular season games
        weekly_game_data = weekly_game_data[weekly_game_data['season_type'] == 'REG']
        
        # Filter for active players
        active_player_ids = set(active_players['player_id'])
        weekly_game_data = weekly_game_data[weekly_game_data['player_id'].isin(active_player_ids)]
        
        # Drop unnecessary columns
        columns_to_drop = ['player_name', 'position_group', 'headshot_url']
        weekly_game_data = weekly_game_data.drop(columns=columns_to_drop)

        # Load the combined game metadata file
        combined_metadata_file = f"{DATA_DIR}/combined_game_metadata.csv"
        game_metadata = pd.read_csv(combined_metadata_file)

        # Perform matching based on season, week, and team involvement
        def enrich_row_with_metadata(row):
            season = row['season']
            week = row['week']
            team1 = row['recent_team']
            team2 = row['opponent_team']

            # Find matching games in the metadata
            match = game_metadata[
                (game_metadata['season'] == season) &
                (game_metadata['week'] == week) &
                (
                    ((game_metadata['home_team'] == team1) & (game_metadata['away_team'] == team2)) |
                    ((game_metadata['home_team'] == team2) & (game_metadata['away_team'] == team1))
                )
            ]

            # Enrich row if a match is found
            if not match.empty:
                enriched_data = match.iloc[0]
                return pd.Series({
                    'game_id': enriched_data['game_id'],
                    'home_team': enriched_data['home_team'],
                    'away_team': enriched_data['away_team'],
                    'home_coach': enriched_data['home_coach'],
                    'away_coach': enriched_data['away_coach'],
                    'winner': enriched_data['winner'],
                    'referee_names': enriched_data['referee_names']
                })
            return pd.Series({
                'game_id': None,
                'home_team': None,
                'away_team': None,
                'home_coach': None,
                'away_coach': None,
                'winner': None,
                'referee_names': None
            })

        # Apply the matching function to enrich the data
        enriched_metadata = weekly_game_data.apply(enrich_row_with_metadata, axis=1)

        # Combine the enriched metadata with the original weekly game data
        weekly_game_data = pd.concat([weekly_game_data, enriched_metadata], axis=1)

        # Log missing game_id entries for debugging
        missing_game_data = weekly_game_data[weekly_game_data['game_id'].isna()]
        if not missing_game_data.empty:
            print(f"Warning: {len(missing_game_data)} rows could not be matched to game metadata.")
            print(missing_game_data[['season', 'week', 'recent_team', 'opponent_team']].head())

        # Fill missing numerical fields with 0 and categorical fields with "Unknown"
        numerical_fields = [
            'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
            'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
            'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'passing_2pt_conversions',
            'pacr', 'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
            'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
            'receiving_fumbles_lost', 'receiving_air_yards', 'receiving_yards_after_catch',
            'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 'racr',
            'target_share', 'air_yards_share', 'wopr', 'special_teams_tds', 'fantasy_points', 'fantasy_points_ppr'
        ]
        weekly_game_data[numerical_fields] = weekly_game_data[numerical_fields].fillna(0)

        categorical_fields = ['game_id', 'home_team', 'away_team', 'home_coach', 'away_coach', 'winner', 'referee_names']
        weekly_game_data[categorical_fields] = weekly_game_data[categorical_fields].fillna("Unknown")

        # Save to CSV
        file_range = f"{year_range[0]}_{year_range[-1]}" if len(year_range) > 1 else f"{year_range[0]}"
        weekly_game_data.to_csv(f"{DATA_DIR}/weekly_game_data_with_ids.csv", index=False)
        print(f"Saved weekly game data with game IDs for {file_range} to {DATA_DIR}/weekly_game_data_with_ids.csv")
        
        return weekly_game_data
    except Exception as e:
        print("Error fetching weekly game data with game IDs:", e)
        raise e



def match_game_metadata(row, game_metadata):
    """
    Match game metadata based on season, week, and teams.
    """
    season = row['season']
    week = row['week']
    team1 = row['recent_team']
    team2 = row['opponent_team']

    # Match game metadata
    match = game_metadata[
        (game_metadata['season'] == season) &
        (game_metadata['week'] == week) &
        (
            ((game_metadata['home_team'] == team1) & (game_metadata['away_team'] == team2)) |
            ((game_metadata['home_team'] == team2) & (game_metadata['away_team'] == team1))
        )
    ]

    if not match.empty:
        enriched_data = match.iloc[0]
        return pd.Series({
            'game_id': enriched_data['game_id'],
            'home_team': enriched_data['home_team'],
            'away_team': enriched_data['away_team'],
            'home_coach': enriched_data['home_coach'],
            'away_coach': enriched_data['away_coach'],
            'winner': enriched_data['winner'],
            'referee_names': enriched_data['referee_names']
        })
    
    # Return empty metadata if no match is found
    return pd.Series({
        'game_id': None,
        'home_team': None,
        'away_team': None,
        'home_coach': None,
        'away_coach': None,
        'winner': None,
        'referee_names': None
    })


def fetch_and_save_combined_game_metadata(year_range):
    try:
        # Combine metadata for all years in the range
        combined_metadata = []

        for year in year_range:
            print(f"Processing game metadata for {year}...")
            pbp_data = nfl.import_pbp_data([year])
            
            # Extract relevant columns for game metadata
            columns_to_keep = ['game_id', 'home_team', 'away_team', 'season', 'week', 'home_coach', 'away_coach']
            game_metadata = pbp_data[columns_to_keep].drop_duplicates(subset=['game_id'])
            
            # Extract the final scores
            final_scores = pbp_data.groupby('game_id').last()[['home_score', 'away_score']].reset_index()
            
            # Merge scores into game metadata
            game_metadata = game_metadata.merge(final_scores, on='game_id', how='left')
            
            # Determine the winner
            game_metadata['winner'] = game_metadata.apply(
                lambda row: row['home_team'] if row['home_score'] > row['away_score'] else (
                    row['away_team'] if row['away_score'] > row['home_score'] else 'TIE'
                ), axis=1
            )
            
            # Fetch the officials data for the specified season
            referee_data = nfl.import_officials([year])
            
            # Aggregate referee data by game_id
            grouped_referees = referee_data.groupby('game_id').agg({
                'name': lambda x: list(x),  # Combine referee names into a list
                'off_pos': lambda x: list(x)  # Combine referee positions into a list
            }).reset_index()
            grouped_referees.rename(columns={'name': 'referee_names', 'off_pos': 'referee_positions'}, inplace=True)
            
            # Merge referee data into game metadata
            game_metadata = game_metadata.merge(grouped_referees, on='game_id', how='left')
            
            # Append to the combined list
            combined_metadata.append(game_metadata)
        
        # Concatenate all metadata into a single DataFrame
        combined_metadata_df = pd.concat(combined_metadata, ignore_index=True)
        
        # Ensure 'season' and 'week' are included
        combined_metadata_df = combined_metadata_df[['game_id', 'home_team', 'away_team', 'season', 'week', 
                                                     'home_coach', 'away_coach', 'home_score', 'away_score', 
                                                     'winner', 'referee_names', 'referee_positions']]
        
        # Save the combined data to a single file
        combined_metadata_file = f"{DATA_DIR}/combined_game_metadata.csv"
        combined_metadata_df.to_csv(combined_metadata_file, index=False)
        print(f"Saved combined game metadata with referees and scores for all years to {combined_metadata_file}")
        
        # Print a sample of the combined data for verification
        print("Sample combined game metadata:")
        print(combined_metadata_df.head())
        
        return combined_metadata_df
    
    except Exception as e:
        print("Error fetching and combining game metadata and referee data:", e)
        raise e



def main():
    # Specify year range
    current_year = 2024
    game_year_range = list(range(2022, 2025)) # this goes up to the 2024

    # Step 1: Fetch active fantasy players
    print("Fetching active fantasy players...")
    active_fantasy_players = fetch_and_save_active_fantasy_players(current_year)
    print("Active fantasy player fetching complete.")

    # Step 2: Fetch combined game metadata (including referees and scores)
    print(f"Fetching combined game metadata for years {game_year_range}...")
    fetch_and_save_combined_game_metadata(game_year_range)
    print("Combined game metadata fetching complete.")

    # Step 3: Fetch weekly game data with game IDs
    print(f"Fetching weekly game data for years {game_year_range}...")
    fetch_and_save_weekly_game_data_with_ids(game_year_range, active_fantasy_players)
    print("Weekly game data fetching complete.")

if __name__ == "__main__":
    main()
