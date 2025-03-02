#!/bin/bash

# Path to the predictions CSV file
PREDICTIONS_FILE="predictions/predictions_week_11_2024.csv"

# Check if the predictions file exists
if [[ ! -f "$PREDICTIONS_FILE" ]]; then
    echo "Error: Predictions file $PREDICTIONS_FILE not found."
    exit 1
fi

# Prompt the user for the game
read -p "Enter the game (e.g., DAL vs HOU): " GAME
HOME_TEAM=$(echo $GAME | awk '{print $1}')
AWAY_TEAM=$(echo $GAME | awk '{print $3}')

# Check if the input is valid
if [[ -z "$HOME_TEAM" || -z "$AWAY_TEAM" ]]; then
    echo "Invalid input. Please specify the game in the format 'TEAM1 vs TEAM2'."
    exit 1
fi

# Filter the players based on the game and output predictions
echo "Player Predictions for $GAME:"
awk -F, -v home="$HOME_TEAM" -v away="$AWAY_TEAM" 'NR==1 || $5 == home || $5 == away { print }' "$PREDICTIONS_FILE" | column -t -s, 

# Extract top performers for each position
echo -e "\nTop Performers by Position:"
POSITIONS=$(awk -F, 'NR>1 { print $4 }' "$PREDICTIONS_FILE" | sort | uniq)

for pos in $POSITIONS; do
    echo "Position: $pos"
    awk -F, -v home="$HOME_TEAM" -v away="$AWAY_TEAM" -v pos="$pos" \
        'NR>1 && ($5 == home || $5 == away) && $4 == pos { print $2, $4, $3, $5, $6 }' "$PREDICTIONS_FILE" | \
        sort -k5 -n -r | head -n 1 | column -t
done
