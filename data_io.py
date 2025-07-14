import os
import pandas as pd
import logging
from glob import glob
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json  
import numpy as np
import os


# Create a platform-independent data directory path
def get_data_dir():
    """Get the appropriate data directory based on environment"""
    # For Docker deployment
    docker_path = "/app/data"
    # For Windows development
    windows_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Check if Docker path exists
    if os.path.exists(docker_path):
        return docker_path
    # Fall back to local development path
    else:
        # Create the data directory if it doesn't exist
        if not os.path.exists(windows_path):
            os.makedirs(windows_path)
        return windows_path

# Get the data directory to use
DATA_DIR = get_data_dir()

def load_match_history_files():
    """Load all match history files in the directory
    
    Returns:
        list: Sorted list of match history file paths
    """
    # Use os.path.join for cross-platform compatibility
    pattern = os.path.join(DATA_DIR, "match_history_with_elo_*.csv")
    files = glob(pattern)
    return sorted(files)

def load_player_data():
    """Load player ELO ratings and statistics
    
    Returns:
        tuple: (DataFrame with player data, filename used)
    """
    try:
        ratings_path = os.path.join(DATA_DIR, "player_elo_ratings.csv")
        df = pd.read_csv(ratings_path)
        return df
    except Exception as e:
        logging.error(f"Error loading player data: {e}")
        return None

def load_session_stats_files():
    """Load all session stats files in the data directory
    
    Returns:
        list: Sorted list of session stats file paths (most recent first)
    """
    # Use os.path.join for cross-platform compatibility
    pattern = os.path.join(DATA_DIR, "session_stats_*.csv")
    files = glob(pattern)
    return sorted(files, reverse=True)  # Most recent first

def read_match_history_file(file_path):
    """Read a specific match history file
    
    Args:
        file_path (str): Path to the match history file
        
    Returns:
        DataFrame: Match history data with session information
    """
    try:
        match_df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        match_df["Session"] = filename.replace("match_history_with_elo_", "").replace(".csv", "")
        match_df['Match_Index'] = range(len(match_df))
        # Split Team A and Team B columns into individual player columns
        match_df[["Player A1", "Player A2"]] = match_df["Team A"].str.split("/", expand=True)
        match_df[["Player B1", "Player B2"]] = match_df["Team B"].str.split("/", expand=True)
        for col in ["Player A1", "Player A2", "Player B1", "Player B2"]:
            match_df[col] = match_df[col].str.strip()
        return match_df
    except Exception as e:
        logging.error(f"Error reading match history file {file_path}: {e}")
        return None

def load_all_match_history():
    """Load all match history data from all files
    
    Returns:
        DataFrame: Combined match history from all files
    """
    files = load_match_history_files()
    all_matches = []
    
    for file in files:
        match_df = read_match_history_file(file)
        if match_df is not None:
            all_matches.append(match_df)
    
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    return None

def read_session_stats(date_str):
    """Read session stats for a specific date
    
    Args:
        date_str (str): Date string in format YYYYMMDD
        
    Returns:
        DataFrame: Session statistics
    """
    session_file = os.path.join(DATA_DIR, f"session_stats_{date_str}.csv")
    
    if os.path.exists(session_file):
        try:
            return pd.read_csv(session_file)
        except Exception as e:
            logging.error(f"Error reading session stats for {date_str}: {e}")
    
    return None

def save_player_data(player_df):
    """Save player data to CSV
    
    Args:
        player_df (DataFrame): Player data to save
        
    Returns:
        bool: Success status
    """
    try:
        filename = "player_elo_ratings.csv"
        file_path = os.path.join(DATA_DIR, filename)
        # backup the old file before saving new data
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(DATA_DIR, f"player_elo_ratings_{current_time}.csv")
        if os.path.exists(file_path):
            os.rename(file_path, backup_path)
        player_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        return True
    except Exception as e:
        logging.error(f"Error saving player data: {e}")
        return False

def save_match_history(match_df, date_str=None):
    """Save match history data for a specific date
    
    Args:
        match_df (DataFrame): Match data to save
        date_str (str, optional): Date string in YYYYMMDD format. If None, today's date is used.
        
    Returns:
        str: Path to the saved file or None if failed
    """
    try:
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
            
        file_path = os.path.join(DATA_DIR, f"match_history_with_elo_{date_str}.csv")
        
        # If file exists, append the new data
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, match_df], ignore_index=True)
            combined_df.to_csv(file_path, index=False, encoding="utf-8-sig")
        else:
            match_df.to_csv(file_path, index=False, encoding="utf-8-sig")
            
        return file_path
    except Exception as e:
        logging.error(f"Error saving match history: {e}")
        return None

def save_session_stats(stats_df, date_str=None):
    """Save session stats for a specific date
    
    Args:
        stats_df (DataFrame): Stats data to save
        date_str (str, optional): Date string in YYYYMMDD format. If None, today's date is used.
        
    Returns:
        bool: Success status
    """
    try:
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
            
        file_path = os.path.join(DATA_DIR, f"session_stats_{date_str}.csv")
        
        # If file exists, update/append the stats
        if os.path.exists(file_path):
            existing_stats = pd.read_csv(file_path)
            
            # For each player in the new stats, update or add their stats
            for _, new_row in stats_df.iterrows():
                player = new_row["Player"]
                
                # Check if player exists in existing stats
                if player in existing_stats["Player"].values:
                    # Update player's stats
                    idx = existing_stats[existing_stats["Player"] == player].index[0]
                    
                    # Special handling for game counters - add them instead of replace
                    # For total games and wins
                    if all(col in existing_stats.columns for col in ['total_games', 'total_wins']):
                        existing_stats.loc[idx, 'total_games'] += new_row['total_games']
                        existing_stats.loc[idx, 'total_wins'] += new_row['total_wins']
                        # Recalculate success rate
                        total_games = existing_stats.loc[idx, 'total_games']
                        total_wins = existing_stats.loc[idx, 'total_wins']
                        if total_games > 0:
                            existing_stats.loc[idx, 'total_success_rate'] = f"{(total_wins / total_games * 100):.1f}%"
                        else:
                            existing_stats.loc[idx, 'total_success_rate'] = "0.0%"
                    
                    # For same gender games and wins
                    if all(col in existing_stats.columns for col in ['same_gender_games', 'same_gender_wins']):
                        existing_stats.loc[idx, 'same_gender_games'] += new_row['same_gender_games']
                        existing_stats.loc[idx, 'same_gender_wins'] += new_row['same_gender_wins']
                        # Recalculate success rate
                        same_gender_games = existing_stats.loc[idx, 'same_gender_games']
                        same_gender_wins = existing_stats.loc[idx, 'same_gender_wins']
                        if same_gender_games > 0:
                            existing_stats.loc[idx, 'same_gender_success_rate'] = f"{(same_gender_wins / same_gender_games * 100):.1f}%"
                        else:
                            existing_stats.loc[idx, 'same_gender_success_rate'] = "0.0%"
                    
                    # For mixed games and wins
                    if all(col in existing_stats.columns for col in ['mixed_games', 'mixed_wins']):
                        existing_stats.loc[idx, 'mixed_games'] += new_row['mixed_games']
                        existing_stats.loc[idx, 'mixed_wins'] += new_row['mixed_wins']
                        # Recalculate success rate
                        mixed_games = existing_stats.loc[idx, 'mixed_games']
                        mixed_wins = existing_stats.loc[idx, 'mixed_wins']
                        if mixed_games > 0:
                            existing_stats.loc[idx, 'mixed_success_rate'] = f"{(mixed_wins / mixed_games * 100):.1f}%"
                        else:
                            existing_stats.loc[idx, 'mixed_success_rate'] = "0.0%"
                    
                    # Update ELO change - add changes cumulatively
                    if 'elo_change' in new_row and 'elo_change' in existing_stats.columns:
                        existing_stats.loc[idx, 'elo_change'] += new_row['elo_change']
                else:
                    # Add new player
                    existing_stats = pd.concat([existing_stats, pd.DataFrame([new_row])], ignore_index=True)
                    
            existing_stats.to_csv(file_path, index=False, encoding="utf-8-sig")
        else:
            stats_df.to_csv(file_path, index=False, encoding="utf-8-sig")
            
        return True
    except Exception as e:
        logging.error(f"Error saving session stats: {e}")
        return False

def add_new_player(player_name, default_elo):
    """Add a new player to the player database
    
    Args:
        player_name (str): Name of the new player
        default_elo (int): Default ELO rating for the new player
        
    Returns:
        bool: Success status
    """
    try:
        # Create new player data row
        stats_columns = [
            "Player", "ELO", "total_games", "total_wins", "total_success_rate",
            "same_gender_games", "same_gender_wins", "same_gender_success_rate",
            "mixed_games", "mixed_wins", "mixed_success_rate"
        ]
        stats_data = [[player_name, default_elo, 0, 0, "0.0%", 0, 0, "0.0%", 0, 0, "0.0%"]]
        new_player_df = pd.DataFrame(stats_data, columns=stats_columns)
        
        # Load existing player data
        player_df = load_player_data()
        
        if player_df is not None:
            # Check if player already exists
            if player_name in player_df["Player"].values:
                return False
            # Add the new player
            combined_df = pd.concat([player_df, new_player_df], ignore_index=True)
            # Save the updated data
            return save_player_data(combined_df)
        else:
            # No existing data, create new file
            return save_player_data(new_player_df)
            
    except Exception as e:
        logging.error(f"Error adding new player: {e}")
        return False


def get_player_match_history(player):
    """Get match history for a specific player"""    
    match_df = load_all_match_history()   
    # Find matches where the player participated
    player_matches = match_df[
                (match_df["Player A1"] == player) | 
                (match_df["Player A2"] == player) | 
                (match_df["Player B1"] == player) | 
                (match_df["Player B2"] == player)
    ].copy()
                    
    # Add columns to identify which team the player was on
    player_matches["Player's Team"] = "A"  # Default to team A
    # Set team to B when player is in Team B
    player_matches.loc[
                (player_matches["Player B1"] == player) | 
                (player_matches["Player B2"] == player),
                "Player's Team"
    ] = "B"
            
    # Determine match result for the player
    player_matches["Result"] = "Won"
    player_matches.loc[
                ((player_matches["Player's Team"] == "A") & (player_matches["Winner"] != "Team A")) |
                ((player_matches["Player's Team"] == "B") & (player_matches["Winner"] != "Team B")),
                "Result"
    ] = "Lost"
    return player_matches.sort_values(by=["Session", "Match_Index"], ascending=[False, False])


def get_player_opponents(player_name):
    """Get all opponents who have played against the given player
    
    Args:
        player_name (str): Name of the player
        
    Returns:
        list: List of opponent player names
    """
    match_df = load_all_match_history()        
    # Find matches where the player participated
    player_matches = match_df[
        (match_df["Player A1"] == player_name) | 
        (match_df["Player A2"] == player_name) | 
        (match_df["Player B1"] == player_name) | 
        (match_df["Player B2"] == player_name)
    ]
    opponents = set()  # Use a set to avoid duplicates  
    # Extract opponents based on which team the player was on
    for _, match in player_matches.iterrows():
        if match["Player A1"] == player_name or match["Player A2"] == player_name:
            # Player was in Team A, so opponents are in Team B
            opponents.add(match["Player B1"])
            opponents.add(match["Player B2"])
        else:
            # Player was in Team B, so opponents are in Team A
            opponents.add(match["Player A1"])
            opponents.add(match["Player A2"])
                        
    # Remove None values and the player's own name if present
    return sorted([opp for opp in opponents if opp and opp != player_name])


def get_head_to_head_history(player1, player2):
    """Get match history between two specific players who played against each other.
    
    Args:
        player1 (str): First player's name
        player2 (str): Second player's name
    
    Returns:
        dict: Dictionary containing summary statistics and match details
    """
    player1_wins = 0
    player2_wins = 0
    match_df = load_all_match_history()        
    # Find matches where player1 and player2 played against each other
    h2h_matches = match_df[
                # player1 in Team A, player2 in Team B
                ((match_df["Player A1"] == player1) & ((match_df["Player B1"] == player2) | (match_df["Player B2"] == player2))) |
                ((match_df["Player A2"] == player1) & ((match_df["Player B1"] == player2) | (match_df["Player B2"] == player2))) |
                # player2 in Team A, player1 in Team B
                ((match_df["Player A1"] == player2) & ((match_df["Player B1"] == player1) | (match_df["Player B2"] == player1))) |
                ((match_df["Player A2"] == player2) & ((match_df["Player B1"] == player1) | (match_df["Player B2"] == player1)))
    ]        
    # Add columns to identify which team each player was on
    h2h_matches["Player1_Team"] = None
    h2h_matches["Player2_Team"] = None
            
    # Determine which team each player was on in each match
    for idx, match in h2h_matches.iterrows():
        if match["Player A1"] == player1 or match["Player A2"] == player1:
            h2h_matches.at[idx, "Player1_Team"] = "A"
            h2h_matches.at[idx, "Player2_Team"] = "B"
        else:
            h2h_matches.at[idx, "Player1_Team"] = "B"
            h2h_matches.at[idx, "Player2_Team"] = "A"
            
    # Count wins for each player
    for idx, match in h2h_matches.iterrows():
        if match["Winner"] == f"Team {match['Player1_Team']}":
            player1_wins += 1
        elif match["Winner"] == f"Team {match['Player2_Team']}":
            player2_wins += 1
            
    sorted_matches = h2h_matches.sort_values(by=["Session", "Match_Index"], ascending=[False, True])
        
    # Create result summary
    summary = {
            "total_matches": len(sorted_matches),
            "player1_name": player1,
            "player2_name": player2,
            "player1_wins": player1_wins,
            "player2_wins": player2_wins,
            "matches": sorted_matches
    }    
    return summary

def get_partnership_statistics():
    """Analyze match history to find partnership statistics
    
    Returns:
        DataFrame: Partnership statistics including games played and win rate
    """
    # Load all match history
    match_df = load_all_match_history()
    
    # Dictionary to track partnership stats
    partnerships = defaultdict(lambda: {"games": 0, "wins": 0})
    
          
    # For each match, extract partnerships and record outcomes
    for _, row in match_df.iterrows():
        team_a_pair = tuple(sorted([row["Player A1"], row["Player A2"]]))    
        team_b_pair = tuple(sorted([row["Player B1"], row["Player B2"]]))
                
        # Update games count for both partnerships
        partnerships[team_a_pair]["games"] += 1
        partnerships[team_b_pair]["games"] += 1
                
        # Update wins based on the winner
        if row["Winner"] == "Team A":
            partnerships[team_a_pair]["wins"] += 1
        elif row["Winner"] == "Team B":
            partnerships[team_b_pair]["wins"] += 1
        
    # Convert the dictionary to a DataFrame for easier display
    if not partnerships:
        return None
        
    partnership_data = []
    for pair, stats in partnerships.items():
        # Calculate win rate
        win_rate = 0 if stats["games"] == 0 else (stats["wins"] / stats["games"]) * 100
        win_rate_str = f"{win_rate:.1f}%"
        
        # Format the partnership name
        partnership_name = f"{pair[0]} / {pair[1]}"
        
        partnership_data.append({
            "Partnership": partnership_name,
            "Games Played": stats["games"],
            "Wins": stats["wins"],
            "Win Rate": win_rate_str,
            "Win Rate Value": win_rate  # Hidden column for sorting
        })
    
    return pd.DataFrame(partnership_data)


def json_matches_to_excel_schedule(input_json_path, output_excel_path, session_id=None, start_time_str=None):
    """
    Process a JSON file with match data (one JSON object per line) and create an Excel schedule.
    Matches are organized from left to right and top to bottom by CompleteTime.
    
    Args:
        input_json_path (str): Path to the JSON file containing match records
        output_excel_path (str): Path where the output Excel file will be saved
        session_id (str, optional): If provided, only matches with this SessionId will be included
        start_time_str (str, optional): Start time for the first round in format "HH:MM"
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set default start time if not provided
        start_time = datetime.strptime("15:00", "%H:%M")
        if start_time_str:
            try:
                start_time = datetime.strptime(start_time_str, "%H:%M")
            except ValueError:
                logging.warning(f"Invalid start time format: {start_time_str}, using default 15:00")
        
        # Read JSON data
        matches = []
        with open(input_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    match = json.loads(line)
                    
                    # Filter by session_id if provided
                    if session_id is not None and match.get('SessionId') != session_id:
                        continue
                        
                    # Parse CompleteTime for sorting
                    if 'CompleteTime' in match:
                        try:
                            match['CompleteTime_dt'] = datetime.strptime(
                                match['CompleteTime'].split('.')[0], 
                                "%Y-%m-%dT%H:%M:%S"
                            )
                        except ValueError:
                            # Try alternative date format
                            match['CompleteTime_dt'] = datetime.strptime(
                                match['CompleteTime'], 
                                "%Y-%m-%dT%H:%M:%S.%fZ"
                            )
                    matches.append(match)
                except Exception as e:
                    logging.error(f"Error parsing JSON line: {e}")
                    continue
        
        if not matches:
            logging.error(f"No valid match data found in the file{' for SessionId ' + session_id if session_id else ''}")
            return False
        
        # Sort matches by CompleteTime
        matches.sort(key=lambda x: x.get('CompleteTime_dt', datetime.min))
        
        
        # Fixed layout with 4 columns for courts
        num_court_columns = 4
        
        # Create a matrix to hold our data
        # First column is for round info, then 4 court columns
        total_cols = num_court_columns + 1
        total_rounds = len(matches) // num_court_columns if len(matches) % num_court_columns == 0 else len(matches) // num_court_columns + 1
        total_rows = total_rounds * 3 + 1 # Each round takes 3 rows (match, empty, score)
        
        # Create empty DataFrame
        data = np.empty((total_rows, total_cols), dtype=object)
        
        # Fill the matrix with match data
        current_row = 0
        for round_num in range(1, total_rounds+1):
            # Calculate timing for this round
            round_start = start_time + timedelta(minutes=(round_num-1)*12)
            round_end = round_start + timedelta(minutes=12)
            
            # Format round timing
            round_time_str = f"Round {int(round_num)} ({round_start.strftime('%H:%M')} - {round_end.strftime('%H:%M')})"
            
            # Add round info to first column
            data[current_row, 0] = round_time_str
            
            # Add matches for this round
            for i, match in enumerate(matches[(round_num-1)*num_court_columns:round_num*num_court_columns]):
                col = (i % num_court_columns) + 1  # +1 because col 0 is for round info
                
                # Calculate row offset within this round block
                row_offset = (i // num_court_columns) * 3
                
                # Extract player data
                player_a1 = match.get('PlayerA1', {}).get('name', '')
                player_a2 = match.get('PlayerA2', {}).get('name', '')
                player_b1 = match.get('PlayerB1', {}).get('name', '')
                player_b2 = match.get('PlayerB2', {}).get('name', '')
                
                # Add gender markers
                if match.get('PlayerA1', {}).get('gender') == 'female':
                    player_a1 += "(F)"
                if match.get('PlayerA2', {}).get('gender') == 'female':
                    player_a2 += "(F)"
                if match.get('PlayerB1', {}).get('gender') == 'female':
                    player_b1 += "(F)"
                if match.get('PlayerB2', {}).get('gender') == 'female':
                    player_b2 += "(F)"
                
                # Format match text and score
                match_text = f"{player_a1}/{player_a2} vs {player_b1}/{player_b2}"
                
                # Format score
                score_a = match.get('ScoreA')
                score_b = match.get('ScoreB')
                score_text = ""
                if score_a is not None and score_b is not None:
                    score_text = f"{score_a}:{score_b}"
                
                # Fill the cells
                data[current_row + row_offset, col] = match_text
                data[current_row + row_offset + 2, col] = score_text
                # Row offset + 1 is left empty for spacing
            
            # Move to next round (leave appropriate spacing)
            current_row += 3
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Add header row
        header = [""] + [f"Court {i}" for i in range(1, num_court_columns + 1)]
        header_df = pd.DataFrame([header])
        
        # Combine header with data
        final_df = pd.concat([header_df, df], ignore_index=True)
        
        # Write to Excel
        final_df.to_excel(output_excel_path, index=False, header=False)
        
        print(f"Excel file created successfully at {output_excel_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error converting JSON to Excel: {e}")
        return False

if __name__ == "__main__":
    json_matches_to_excel_schedule(r"C:\Users\qiaominye\Downloads\database_export-elo-system-8g6jq2r4a931945e-Match.json", '20250712.xlsx', 'game1752030905605', "17:00")
