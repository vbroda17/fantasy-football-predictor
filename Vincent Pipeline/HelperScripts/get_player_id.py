import pandas as pd

# File paths
ACTIVE_PLAYERS_FILE = 'nfl_data/active_fantasy_players.csv'
PLAYER_INFO_FILE = 'nfl_data/player_info.csv'

def main():
    # Read the data
    active_players = pd.read_csv(ACTIVE_PLAYERS_FILE)
    
    # Ensure the necessary columns are present
    required_columns = ['player_name', 'team', 'position', 'player_id']
    alternative_columns = {
        'team': 'recent_team'  # Map 'team' to 'recent_team' if 'team' doesn't exist
    }
    
    for i, col in enumerate(required_columns):
        if col not in active_players.columns:
            if col in alternative_columns:
                alt_col = alternative_columns[col]
                if alt_col in active_players.columns:
                    required_columns[i] = alt_col
                else:
                    raise ValueError(f"Column '{col}' or its alternative '{alt_col}' not found in the data.")
            else:
                raise ValueError(f"Column '{col}' not found in the data.")
    
    # Extract the required columns
    player_info = active_players[required_columns]
    # Rename columns to standard names
    player_info.columns = ['player_name', 'team', 'position', 'player_id']
    
    # Save to CSV
    player_info.to_csv(PLAYER_INFO_FILE, index=False)
    print(f"Saved player information to {PLAYER_INFO_FILE}")
    
    # Interactive lookup. Uncomment this part of the code out if you want to look up player IDs
    # while True:
    #     player_name_to_lookup = input("Enter the player's name to look up (or 'exit' to quit): ")
    #     if player_name_to_lookup.lower() == 'exit':
    #         break
    #     matches = lookup_player(player_name_to_lookup, player_info)
    #     if matches is not None:
    #         print(matches)
    #         print()
    
def lookup_player(name, player_info_df):
    """
    Look up players by name.
    Returns a DataFrame with matching player(s).
    """
    # Case-insensitive search
    matches = player_info_df[player_info_df['player_name'].str.lower() == name.lower()]
    
    if matches.empty:
        print(f"No players found with the name '{name}'.")
        return None
    else:
        print(f"Found {len(matches)} player(s) with the name '{name}':")
        return matches

if __name__ == "__main__":
    main()
