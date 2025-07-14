import pandas as pd
from collections import defaultdict
from data_io import save_player_data, save_match_history, save_session_stats

DEFAULT_ELO = 1500  # 默认 Elo 分数

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

def load_existing_player_data():
    """
    Load existing player ELO ratings and statistics from CSV.
    
    Args:
        data_dir: Directory to look for existing data files
        
    Returns:
        Tuple containing player ELO ratings dict and player statistics dict
    """
    from data_io import load_player_data
    player_stats = defaultdict(lambda: {
        "total": {"games": 0, "wins": 0},
        "male_double": {"games": 0, "wins": 0},
        "female_double": {"games": 0, "wins": 0},
        "mixed": {"games": 0, "wins": 0}
    })
    try:
        existing_df = load_player_data()
        print(f"Loaded data for {len(existing_df)} existing players")
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
        return player_elos, player_stats
        
    except FileNotFoundError:
        print(f"No existing data found. All players will start with {DEFAULT_ELO} ELO and no stats.")
        return defaultdict(lambda: DEFAULT_ELO), player_stats
    except Exception as e:
        print(f"Error loading existing data: {e}. All players will start with {DEFAULT_ELO} ELO and no stats.")
        return defaultdict(lambda: DEFAULT_ELO), player_stats


def extract_matches_from_excel(excel_path, sheet_name="Schedule"):
    """Extract match data and timing information from Excel file."""
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
    
    # Get court names from the first row of the Excel file
    court_names = {}
    for col in cols:
        if 0 < df.shape[0] and col < df.shape[1]:
            header_value = str(df.iat[0, col]) if not pd.isna(df.iat[0, col]) else f"Court {col}"
            court_names[col] = header_value
    print(f"Court names from Excel: {court_names}")
    
    # Extract round information and timing from the first column
    rounds_info = {}
    for row in range(df.shape[0]):
        if row < df.shape[0] and 0 < df.shape[1]:
            cell_value = str(df.iat[row, 0]) if not pd.isna(df.iat[row, 0]) else ""
            if "Round" in cell_value and "(" in cell_value and ")" in cell_value:
                try:
                    # Extract round number
                    round_num = float(cell_value.split("Round")[1].split("(")[0].strip())
                    
                    # Extract timing information
                    time_info = cell_value.split("(")[1].split(")")[0].strip()
                    start_time, end_time = [t.strip() for t in time_info.split("-")]
                    
                    # Store round information
                    rounds_info[row] = {
                        "round_num": round_num,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                except Exception as e:
                    print(f"Error parsing round information from '{cell_value}': {e}")
    
    matches = []
    current_round_info = None
    
    # Extract match details from the identified columns
    for row in range(1, df.shape[0] - 1, 3):
        # Check if this row starts a new round
        if row in rounds_info:
            current_round_info = rounds_info[row]
            
        for col in cols:
            if row < df.shape[0] and col < df.shape[1]:
                match_text = df.iat[row, col] if not pd.isna(df.iat[row, col]) else None
                score_text = df.iat[row + 2, col] if row + 2 < df.shape[0] and not pd.isna(df.iat[row + 2, col]) else None
                if match_text and score_text:
                    # Use court name from the first row, or fall back to "Court {col}" if not available
                    court_name = court_names.get(col, f"Court {col}")
                    # Store as (court, match, score, round_info)
                    matches.append((court_name, str(match_text), str(score_text), current_round_info))
    
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
        
        # Store pre-match individual ELO ratings
        pre_elos = {
            left_team[0]: player_elos[left_team[0]],
            left_team[1]: player_elos[left_team[1]],
            right_team[0]: player_elos[right_team[0]],
            right_team[1]: player_elos[right_team[1]]
        }
        
        # Calculate pre-match team ELO for display
        team_a_pre_elo = compute_team_elo(player_elos[left_team[0]], player_elos[left_team[1]])
        team_b_pre_elo = compute_team_elo(player_elos[right_team[0]], player_elos[right_team[1]])
        
        # Update ELO ratings
        update_doubles_match(
            team_a=(left_team[0], left_team[1]),
            team_b=(right_team[0], right_team[1]),
            elos=player_elos,
            winner=winner
        )
        
        # Store post-match individual ELO ratings
        post_elos = {
            left_team[0]: player_elos[left_team[0]],
            left_team[1]: player_elos[left_team[1]],
            right_team[0]: player_elos[right_team[0]],
            right_team[1]: player_elos[right_team[1]]
        }
        
        # Calculate post-match team ELO for display
        team_a_post_elo = compute_team_elo(player_elos[left_team[0]], player_elos[left_team[1]])
        team_b_post_elo = compute_team_elo(player_elos[right_team[0]], player_elos[right_team[1]])
        
        # Record match details with individual player ELO scores
        match_record = {
            'Court': court,
            'Team A': f"{left_team[0]}/{left_team[1]}",
            'Team B': f"{right_team[0]}/{right_team[1]}",
            'Score': f"{score_left}:{score_right}",
            'Winner': 'Team A' if winner == 'A' else 'Team B',
            'ELO Change A': team_a_post_elo - team_a_pre_elo,
            'ELO Change B': team_b_post_elo - team_b_pre_elo,
            # Individual player ELO scores
            'PlayerA1 Pre-ELO': pre_elos[left_team[0]],
            'PlayerA1 Post-ELO': post_elos[left_team[0]],
            'PlayerA2 Pre-ELO': pre_elos[left_team[1]],
            'PlayerA2 Post-ELO': post_elos[left_team[1]],
            'PlayerB1 Pre-ELO': pre_elos[right_team[0]],
            'PlayerB1 Post-ELO': post_elos[right_team[0]],
            'PlayerB2 Pre-ELO': pre_elos[right_team[1]],
            'PlayerB2 Post-ELO': post_elos[right_team[1]]
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


def calculate_elo_ratings(excel_path, sheet_name="Schedule"):
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
    player_elos, player_stats = load_existing_player_data()
    
    # Store initial ELO ratings to calculate changes after the session
    initial_elos = dict(player_elos)
    
    # Extract matches from Excel file
    matches = extract_matches_from_excel(excel_path, sheet_name)
    
    # Process matches and update ELO ratings and statistics
    match_history = []
    for court, match, score, _ in matches:
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
        
    combined_df = pd.merge(elo_df, stats_df, on="Player")
    save_player_data(combined_df)
    save_match_history(history_df)
    save_session_stats(session_df)  
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
    elo_df, history_df, stats_df, session_df = calculate_elo_ratings('./results/20250712.xlsx', "Sheet1")
