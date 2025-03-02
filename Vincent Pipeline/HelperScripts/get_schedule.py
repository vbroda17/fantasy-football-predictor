
import nfl_data_py as nfl
import pandas as pd
import os

# Directory to save data files
DATA_DIR = "nfl_data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_and_save_schedules(years):
    """
    Fetches the NFL schedules for the specified years and saves them to a CSV file.
    """
    try:
        print(f"Fetching NFL schedules for years: {years}...")
        # Fetch the schedules
        schedules = nfl.import_schedules(years=years)
        
        # Print available columns
        print("Available columns in schedules DataFrame:")
        print(schedules.columns.tolist())

        # Define the desired columns
        desired_columns = [
            'season', 'season_type', 'week', 'game_id',
            'game_date', 'game_time_eastern',
            'home_team', 'away_team', 'site_city', 'site_state', 'result'
        ]

        # Keep only columns that are available
        available_columns = schedules.columns.tolist()
        columns_to_keep = [col for col in desired_columns if col in available_columns]

        schedules = schedules[columns_to_keep]

        # Convert 'game_date' to datetime if it's available
        if 'game_date' in schedules.columns:
            schedules['game_date'] = pd.to_datetime(schedules['game_date'])

        # Sort the schedules by available columns
        sort_columns = ['season', 'week']
        if 'game_date' in schedules.columns:
            sort_columns.append('game_date')
        schedules.sort_values(by=sort_columns, inplace=True)

        # Save to CSV without years in the filename
        schedules_file = os.path.join(DATA_DIR, 'nfl_schedules.csv')
        schedules.to_csv(schedules_file, index=False)
        print(f"Saved NFL schedules to {schedules_file}")
        
        # Print a sample of the data
        print("Sample NFL schedules data:")
        print(schedules.head())

        return schedules

    except Exception as e:
        print("Error fetching NFL schedules:", e)
        raise e

def main():
    # Specify the years for which you want to fetch schedules
    years_to_fetch = [2024]  # Only 2024 as per your request
    
    # Fetch and save the schedules
    fetch_and_save_schedules(years_to_fetch)

if __name__ == "__main__":
    main()
