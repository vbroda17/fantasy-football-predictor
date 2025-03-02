import pandas as pd

def extract_and_save_head_coaches(input_file, output_file="nfl_data/team_head_coaches.csv"):
    """
    Extract head coaches for each team and save them to a CSV file.
    
    Parameters:
        input_file (str): Path to the input file containing game data.
        output_file (str): Path to the output CSV file.
    """
    print("Extracting head coaches data...")
    
    # Read the input data
    data = pd.read_csv(input_file)
    
    # Check for necessary columns
    required_columns = ['home_team', 'home_coach', 'away_team', 'away_coach']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Input file must contain the following columns: {required_columns}")
    
    # Create a dictionary to store unique team-coach pairs
    team_coaches = {}

    for _, row in data.iterrows():
        if pd.notna(row['home_coach']):
            team_coaches[row['home_team']] = row['home_coach']
        if pd.notna(row['away_coach']):
            team_coaches[row['away_team']] = row['away_coach']

    # Convert the dictionary to a DataFrame
    coaches_df = pd.DataFrame(team_coaches.items(), columns=['team', 'head_coach']).sort_values(by='team')

    # Save to CSV
    print(f"Saving head coaches data to {output_file}...")
    coaches_df.to_csv(output_file, index=False)
    print("Head coaches data saved successfully.")

# Example usage
input_file = "nfl_data/weekly_game_data_with_ids.csv"
output_file = "nfl_data/team_head_coaches.csv"
extract_and_save_head_coaches(input_file, output_file)
