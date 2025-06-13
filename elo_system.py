# 羽毛球双打 Elo 分系统
import pandas as pd
from collections import defaultdict
from datetime import datetime
import os

DEFAULT_ELO = 1550  # 默认 Elo 分数

def expected_score(team_elo_a, team_elo_b):
    """计算 A 队胜率预期"""
    return 1 / (1 + 10 ** ((team_elo_b - team_elo_a) / 400))

def update_elo(player_elo, expected, actual, k=32):
    """更新某位选手 Elo 分"""
    return player_elo + k * (actual - expected)

def compute_team_elo(player1_elo, player2_elo):
    """队伍 Elo 平均"""
    return (player1_elo + player2_elo) / 2

def update_doubles_match(
    team_a,  # (p1_name, p2_name)
    team_b,
    elos,    # {'player': elo_score}
    winner   # 'A' or 'B'
):
    p1a, p2a = team_a
    p1b, p2b = team_b

    team_a_elo = compute_team_elo(elos[p1a], elos[p2a])
    team_b_elo = compute_team_elo(elos[p1b], elos[p2b])

    expected_a = expected_score(team_a_elo, team_b_elo)
    expected_b = 1 - expected_a

    actual_a = 1.0 if winner == 'A' else 0.0
    actual_b = 1.0 - actual_a

    # 更新每个选手的 Elo
    elos[p1a] = update_elo(elos[p1a], expected_a, actual_a)
    elos[p2a] = update_elo(elos[p2a], expected_a, actual_a)
    elos[p1b] = update_elo(elos[p1b], expected_b, actual_b)
    elos[p2b] = update_elo(elos[p2b], expected_b, actual_b)
    return elos

def load_existing_player_data(data_dir=None):
    """
    Load existing player ELO ratings and statistics from CSV.
    
    Args:
        data_dir: Directory to look for existing data files
        
    Returns:
        Tuple containing player ELO ratings dict and player statistics dict
    """
    
    player_stats = defaultdict(lambda: {
        "total": {"games": 0, "wins": 0},
        "male_double": {"games": 0, "wins": 0},
        "female_double": {"games": 0, "wins": 0},
        "mixed": {"games": 0, "wins": 0}
    })
    
    # Use data_dir if provided
    ratings_file = "player_elo_ratings.csv"
    if data_dir:
        ratings_file = os.path.join(data_dir, ratings_file)
    
    try:
        existing_df = pd.read_csv(ratings_file)
        print(f"Loaded data for {len(existing_df)} existing players from {ratings_file}")
        
        # Create a dictionary from Player -> ELO column
        existing_elos = dict(zip(existing_df['Player'], existing_df['ELO']))
        
        # Initialize ELO dictionary with existing values, defaulting to DEFAULT_ELO for new players
        player_elos = defaultdict(lambda: DEFAULT_ELO)
        # Update with existing values
        for player, elo in existing_elos.items():
            player_elos[player] = elo
            
        # Load existing statistics for each player
        for _, row in existing_df.iterrows():
            player = row['Player']
            
            # Check for required columns
            if all(col in existing_df.columns for col in ['total_games', 'total_wins']):
                # Load total stats
                player_stats[player]["total"]["games"] = row['total_games']
                player_stats[player]["total"]["wins"] = row['total_wins']
                
                # Determine gender and corresponding stats columns
                is_female = player.endswith('(F)')
                gender_specific_type = "female_double" if is_female else "male_double"
                
                # Load same gender stats
                if all(col in existing_df.columns for col in ['same_gender_games', 'same_gender_wins']):
                    player_stats[player][gender_specific_type]["games"] = row['same_gender_games']
                    player_stats[player][gender_specific_type]["wins"] = row['same_gender_wins']
                
                # Load mixed stats
                if all(col in existing_df.columns for col in ['mixed_games', 'mixed_wins']):
                    player_stats[player]["mixed"]["games"] = row['mixed_games']
                    player_stats[player]["mixed"]["wins"] = row['mixed_wins']
        print(f"Loaded statistics and ratings for existing players")
        player_elos['浩南'] = 1600
        player_elos['OwenWei'] = 1650 
        player_elos['方文'] = 1600 
        return player_elos, player_stats
        
    except FileNotFoundError:
        print(f"No existing data found. All players will start with {DEFAULT_ELO} ELO and no stats.")
        return defaultdict(lambda: DEFAULT_ELO), player_stats
    except Exception as e:
        print(f"Error loading existing data: {e}. All players will start with {DEFAULT_ELO} ELO and no stats.")
        return defaultdict(lambda: DEFAULT_ELO), player_stats


def extract_matches_from_excel(excel_path, sheet_name="Schedule"):
    """Extract match data from Excel file."""
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # Auto-detect columns with match data
    cols = []
    # Check each column for match pattern (looking for " vs " in cells)
    for col in range(df.shape[1]):
        # Check if this column contains any match data
        has_matches = False
        for row in range(1, df.shape[0] - 1, 3):
            if row < df.shape[0] and col < df.shape[1]:
                cell_value = str(df.iat[row, col]) if not pd.isna(df.iat[row, col]) else ""
                if " vs " in cell_value:
                    has_matches = True
                    break
        if has_matches:
            cols.append(col)
    print(f"Auto-detected columns with match data: {cols}")
    
    matches = []
    # Extract match details from the identified columns
    for row in range(1, df.shape[0] - 1, 3):
        for col in cols:
            if row < df.shape[0] and col < df.shape[1]:
                match_text = df.iat[row, col] if not pd.isna(df.iat[row, col]) else None
                score_text = df.iat[row + 2, col] if row + 2 < df.shape[0] and not pd.isna(df.iat[row + 2, col]) else None
                if match_text and score_text:
                    # Store as (court, match, score)
                    matches.append((f"Court {col}", str(match_text), str(score_text)))
    
    print(f"Extracted {len(matches)} matches from {len(cols)} columns")
    return matches


def determine_team_type(team):
    """Determine the game type based on team composition."""
    is_female = [player.endswith('(F)') for player in team]
    if all(is_female):
        return "female_double"
    elif not any(is_female):
        return "male_double"
    else:
        return "mixed"


def process_match(court, match, score, player_elos, player_stats, session_stats):
    """Process a single match and update ELO ratings and statistics."""
    try:
        # Parse score
        score_parts = score.strip().split(":")
        if len(score_parts) < 2:
            print(f"Skipping match with invalid score format: {score}")
            return None
            
        score_left = int(score_parts[0])
        score_right = int(score_parts[1])
        
        # Parse teams
        teams = match.split(" vs ")
        if len(teams) != 2:
            print(f"Skipping match with invalid format: {match}")
            return None

        left_team = [player.strip() for player in teams[0].split("/")]
        right_team = [player.strip() for player in teams[1].split("/")]
        
        if len(left_team) != 2 or len(right_team) != 2:
            print(f"Skipping match without 2 players per team: {match}")
            return None
        
        # Determine winner
        winner = 'A' if score_left > score_right else 'B'
        
        # Calculate pre-match ELO for tracking
        team_a_pre_elo = compute_team_elo(player_elos[left_team[0]], player_elos[left_team[1]])
        team_b_pre_elo = compute_team_elo(player_elos[right_team[0]], player_elos[right_team[1]])
        
        # Update ELO ratings
        update_doubles_match(
            team_a=(left_team[0], left_team[1]),
            team_b=(right_team[0], right_team[1]),
            elos=player_elos,
            winner=winner
        )
        
        # Calculate ELO changes for history
        team_a_post_elo = compute_team_elo(player_elos[left_team[0]], player_elos[left_team[1]])
        team_b_post_elo = compute_team_elo(player_elos[right_team[0]], player_elos[right_team[1]])
        
        # Record match details and ELO changes
        match_record = {
            'Court': court,
            'Team A': f"{left_team[0]}/{left_team[1]}",
            'Team B': f"{right_team[0]}/{right_team[1]}",
            'Score': f"{score_left}:{score_right}",
            'Winner': 'Team A' if winner == 'A' else 'Team B',
            'Team A Pre-ELO': round(team_a_pre_elo, 1),
            'Team B Pre-ELO': round(team_b_pre_elo, 1),
            'Team A Post-ELO': round(team_a_post_elo, 1),
            'Team B Post-ELO': round(team_b_post_elo, 1),
            'ELO Change A': round(team_a_post_elo - team_a_pre_elo, 1),
            'ELO Change B': round(team_b_post_elo - team_b_pre_elo, 1)
        }
        
        # Update individual player statistics by game type
        winning_team = set(left_team) if score_left > score_right else set(right_team)
        
        # Determine game types for each team
        team_a_type = determine_team_type(left_team)
        team_b_type = determine_team_type(right_team)
        
        # Update stats for Team A
        for player in left_team:
            # Determine the game type for this player
            game_type = team_a_type
            
            # Update total stats
            session_stats[player]["total"]["games"] += 1
            player_stats[player]["total"]["games"] += 1
            player_stats[player][game_type]["games"] += 1
            session_stats[player][game_type]["games"] += 1
            if player in winning_team:
                player_stats[player]["total"]["wins"] += 1
                session_stats[player]["total"]["wins"] += 1
                player_stats[player][game_type]["wins"] += 1
                session_stats[player][game_type]["wins"] += 1
                
        # Update stats for Team B
        for player in right_team:
            # Determine the game type for this player
            game_type = team_b_type
            
            # Update total stats
            player_stats[player]["total"]["games"] += 1
            player_stats[player][game_type]["games"] += 1
            session_stats[player]["total"]["games"] += 1
            session_stats[player][game_type]["games"] += 1
            if player in winning_team:
                player_stats[player]["total"]["wins"] += 1
                player_stats[player][game_type]["wins"] += 1
                session_stats[player]["total"]["wins"] += 1
                session_stats[player][game_type]["wins"] += 1
                
        return match_record
        
    except Exception as e:
        print(f"Error processing match {match} with score {score}: {e}")
        return None


def create_stats_dataframe(stats_dict):
    """Convert player statistics dictionary to a DataFrame."""
    stats_rows = []
    
    for player, stats in stats_dict.items():
        row = {'Player': player}
        
        # Add total games and wins
        row['total_games'] = stats['total']['games']
        row['total_wins'] = stats['total']['wins']
        row['total_success_rate'] = calculate_success_rate(stats['total']['wins'], stats['total']['games'])
        
        # Determine which doubles type to use for this player based on gender
        is_female = player.endswith('(F)')
        gender_specific_type = "female_double" if is_female else "male_double"
        
        # Add same-gender doubles stats
        row['same_gender_games'] = stats[gender_specific_type]['games']
        row['same_gender_wins'] = stats[gender_specific_type]['wins']
        row['same_gender_success_rate'] = calculate_success_rate(stats[gender_specific_type]['wins'], 
                                                               stats[gender_specific_type]['games'])
        
        # Add mixed stats
        row['mixed_games'] = stats['mixed']['games']
        row['mixed_wins'] = stats['mixed']['wins']
        row['mixed_success_rate'] = calculate_success_rate(stats['mixed']['wins'], stats['mixed']['games'])
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df.sort_values(by=['total_wins', 'total_games'], ascending=False)
    
    return stats_df


def save_results(elo_df, stats_df, history_df, session_df, data_dir=None):
    """
    Save all results to CSV files with appropriate timestamps.
    
    Args:
        elo_df: DataFrame containing player ELO ratings
        stats_df: DataFrame containing player statistics
        history_df: DataFrame containing match history
        session_df: DataFrame containing current session statistics
        data_dir: Directory to save files to (defaults to current directory if None)
    """
    
    # Use data_dir if provided, otherwise use current directory
    save_dir = data_dir if data_dir else "."
    
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Merge player stats with ELO ratings
    combined_df = pd.merge(elo_df, stats_df, on="Player")
    
    # Save combined results to player_elo_ratings_new.csv
    combined_path = os.path.join(save_dir, "player_elo_ratings_new.csv")
    combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
    
    # Generate timestamp for the match history file
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save match history with timestamp in filename
    match_history_filename = f"match_history_with_elo_{timestamp}.csv"
    history_path = os.path.join(save_dir, match_history_filename)
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    
    # Save session stats with timestamp in filename
    session_stats_filename = f"session_stats_{timestamp}.csv"
    session_path = os.path.join(save_dir, session_stats_filename)
    if not session_df.empty:
        session_df.to_csv(session_path, index=False, encoding="utf-8-sig")
    
    print(f"\nCombined ELO ratings and statistics saved to '{combined_path}'")
    print(f"Match history with ELO changes saved to '{history_path}'")
    print(f"Current session statistics saved to '{session_path}'")


def calculate_elo_ratings(excel_path, sheet_name="Schedule", data_dir=None):
    """
    Calculate ELO ratings for players based on match results in the Excel file.
    Uses existing ratings and statistics from player_elo_ratings.csv if available.
    
    Args:
        excel_path: Path to the Excel file containing match results
        sheet_name: Name of the sheet to read from Excel file
        data_dir: Directory to save files to and look for existing data
        
    Returns:
        Tuple containing DataFrames for ELO ratings, match history, player statistics, and session statistics
    """
    # Initialize session stats tracker
    session_stats = defaultdict(lambda: {
        "total": {"games": 0, "wins": 0},
        "male_double": {"games": 0, "wins": 0},
        "female_double": {"games": 0, "wins": 0},
        "mixed": {"games": 0, "wins": 0}
    })
    
    # Load existing player data
    player_elos, player_stats = load_existing_player_data(data_dir)
    
    # Store initial ELO ratings to calculate changes after the session
    initial_elos = dict(player_elos)
    
    # Extract matches from Excel file
    matches = extract_matches_from_excel(excel_path, sheet_name)
    
    # Process matches and update ELO ratings and statistics
    match_history = []
    for court, match, score in matches:
        match_record = process_match(
            court, match, score,
            player_elos, player_stats, session_stats
        )
        if match_record:
            match_history.append(match_record)
    
    # Convert ELO dictionary to DataFrame for display
    elo_df = pd.DataFrame(player_elos.items(), columns=['Player', 'ELO'])
    elo_df = elo_df.sort_values(by='ELO', ascending=False).reset_index(drop=True)
    
    # Create DataFrames for match history and player statistics
    history_df = pd.DataFrame(match_history)
    stats_df = create_stats_dataframe(player_stats)
    session_df = create_stats_dataframe(session_stats)
    
    # Calculate ELO changes for this session
    elo_changes = {}
    for player, final_elo in player_elos.items():
        # If player is new (not in initial_elos), initial ELO was DEFAULT_ELO
        initial_elo = initial_elos.get(player, DEFAULT_ELO)
        elo_changes[player] = round(final_elo - initial_elo, 1)
    
    # Add ELO change column to session_df
    session_df['elo_change'] = session_df['Player'].map(elo_changes)
    
    # Print final ELO ratings
    print("\n=== FINAL ELO RATINGS ===")
    print(elo_df.to_string(float_format='%.1f'))
    
    # Save all results to files
    save_results(elo_df, stats_df, history_df, session_df, data_dir)
    
    return elo_df, history_df, stats_df, session_df

def calculate_success_rate(wins, games):
    """Calculate success rate as a formatted percentage string"""
    if games == 0:
        return "0.0%"
    rate = (wins / games) * 100
    return f"{rate:.1f}%"

if __name__ == "__main__":
    # Example usage:
    # Specify the path to your Excel file containing match results
    # You can change this path to your actual file location
    elo_df, history_df, stats_df, session_df = calculate_elo_ratings('./results/20250604.xlsx', data_dir='data')
