# 羽毛球双打 TrueSkill 分系统
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import os
import trueskill
from trueskill import Rating, rate

# Import reusable functions from elo_system
from elo_system import (
    extract_matches_from_excel, 
    determine_team_type, 
    create_stats_dataframe
)

# Initialize the TrueSkill environment
# These parameters can be adjusted based on your preference
ts_env = trueskill.TrueSkill(
    mu=1500,         # Initial mean value (equivalent to starting ELO)
    sigma=250,       # Initial standard deviation (uncertainty)
    beta=125,        # Skill distance that guarantees ~76% win probability
    tau=5,           # Dynamic factor, adjusts for rating volatility over time
    draw_probability=0.0  # No draws in badminton
)

def compute_win_probability(team_a, team_b):
    """Calculate probability that team_a will win against team_b"""
    delta_mu = sum(r.mu for r in team_a) - sum(r.mu for r in team_b)
    sum_sigma = sum(r.sigma ** 2 for r in team_a) + sum(r.sigma ** 2 for r in team_b)
    denominator = np.sqrt(len(team_a) * len(team_b) * ts_env.beta ** 2 + sum_sigma)
    return ts_env.cdf(delta_mu / denominator)

def update_doubles_match(
    team_a,  # (p1_name, p2_name)
    team_b,
    ratings,  # {'player': Rating(mu, sigma)}
    winner    # 'A' or 'B'
):
    """Update player ratings after a doubles match"""
    p1a, p2a = team_a
    p1b, p2b = team_b
    
    # Get team ratings
    team_a_ratings = [ratings[p1a], ratings[p2a]]
    team_b_ratings = [ratings[p1b], ratings[p2b]]
    
    # Calculate win probability for reporting
    team_a_win_prob = compute_win_probability(team_a_ratings, team_b_ratings)
    team_b_win_prob = 1 - team_a_win_prob
    
    # Record pre-match ratings
    pre_ratings = {
        p1a: Rating(mu=ratings[p1a].mu, sigma=ratings[p1a].sigma),
        p2a: Rating(mu=ratings[p2a].mu, sigma=ratings[p2a].sigma),
        p1b: Rating(mu=ratings[p1b].mu, sigma=ratings[p1b].sigma),
        p2b: Rating(mu=ratings[p2b].mu, sigma=ratings[p2b].sigma)
    }
    
    # Update ratings based on match result
    if winner == 'A':
        (new_a1, new_a2), (new_b1, new_b2) = ts_env.rate([team_a_ratings, team_b_ratings], ranks=[0, 1])
    else:  # B won
        (new_a1, new_a2), (new_b1, new_b2) = ts_env.rate([team_a_ratings, team_b_ratings], ranks=[1, 0])
    
    # Store updated ratings
    ratings[p1a] = new_a1
    ratings[p2a] = new_a2
    ratings[p1b] = new_b1
    ratings[p2b] = new_b2
    
    # Create return object with rating changes
    changes = {
        p1a: ratings[p1a].mu - pre_ratings[p1a].mu,
        p2a: ratings[p2a].mu - pre_ratings[p2a].mu,
        p1b: ratings[p1b].mu - pre_ratings[p1b].mu,
        p2b: ratings[p2b].mu - pre_ratings[p2b].mu
    }
    
    result = {
        'team_a_win_prob': team_a_win_prob,
        'team_b_win_prob': team_b_win_prob,
        'pre_ratings': pre_ratings,
        'post_ratings': {
            p1a: ratings[p1a],
            p2a: ratings[p2a],
            p1b: ratings[p1b],
            p2b: ratings[p2b]
        },
        'changes': changes
    }
    
    return ratings, result

def load_existing_player_data(data_dir=None):
    """
    Load existing player TrueSkill ratings and statistics from CSV.
    
    Args:
        data_dir: Directory to look for existing data files
        
    Returns:
        Tuple containing player TrueSkill ratings dict and player statistics dict
    """
    
    player_stats = defaultdict(lambda: {
        "total": {"games": 0, "wins": 0},
        "male_double": {"games": 0, "wins": 0},
        "female_double": {"games": 0, "wins": 0},
        "mixed": {"games": 0, "wins": 0}
    })
    
    # Use data_dir if provided
    ratings_file = "player_trueskill_ratings.csv"
    if data_dir:
        ratings_file = os.path.join(data_dir, ratings_file)
    
    try:
        existing_df = pd.read_csv(ratings_file)
        print(f"Loaded data for {len(existing_df)} existing players from {ratings_file}")
        
        # Initialize ratings dictionary with default values for new players
        player_ratings = defaultdict(lambda: Rating(mu=ts_env.mu, sigma=ts_env.sigma))
        
        # Update with existing values
        for _, row in existing_df.iterrows():
            player = row['Player']
            # Create Rating object with loaded mu and sigma values
            player_ratings[player] = Rating(mu=row['Mean'], sigma=row['Sigma'])
            
            # Load statistics if they exist in the file
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
        return player_ratings, player_stats
        
    except FileNotFoundError:
        print("No existing data found. All players will start with default TrueSkill ratings.")
        return defaultdict(lambda: Rating(mu=ts_env.mu, sigma=ts_env.sigma)), player_stats
    except Exception as e:
        print(f"Error loading existing data: {e}. All players will start with default TrueSkill ratings.")
        return defaultdict(lambda: Rating(mu=ts_env.mu, sigma=ts_env.sigma)), player_stats


def process_match(court, match, score, player_ratings, player_stats, session_stats):
    """Process a single match and update TrueSkill ratings and statistics."""
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
        
        # Calculate pre-match ratings for tracking
        team_a_pre_rating = [player_ratings[left_team[0]], player_ratings[left_team[1]]]
        team_b_pre_rating = [player_ratings[right_team[0]], player_ratings[right_team[1]]]
        
        # Calculate pre-match win probabilities
        team_a_win_prob = compute_win_probability(team_a_pre_rating, team_b_pre_rating)
        
        # Store pre-match mean ratings for reporting
        team_a_pre_mu = (player_ratings[left_team[0]].mu + player_ratings[left_team[1]].mu) / 2
        team_b_pre_mu = (player_ratings[right_team[0]].mu + player_ratings[right_team[1]].mu) / 2
        
        # Update TrueSkill ratings
        _, rating_details = update_doubles_match(
            team_a=(left_team[0], left_team[1]),
            team_b=(right_team[0], right_team[1]),
            ratings=player_ratings,
            winner=winner
        )
        
        # Calculate post-match mean ratings
        team_a_post_mu = (player_ratings[left_team[0]].mu + player_ratings[left_team[1]].mu) / 2
        team_b_post_mu = (player_ratings[right_team[0]].mu + player_ratings[right_team[1]].mu) / 2
        
        # Record match details and rating changes
        match_record = {
            'Court': court,
            'Team A': f"{left_team[0]}/{left_team[1]}",
            'Team B': f"{right_team[0]}/{right_team[1]}",
            'Score': f"{score_left}:{score_right}",
            'Winner': 'Team A' if winner == 'A' else 'Team B',
            'Team A Win Prob': round(team_a_win_prob * 100, 1),
            'Team B Win Prob': round((1 - team_a_win_prob) * 100, 1),
            'Team A Pre-Rating': round(team_a_pre_mu, 1),
            'Team B Pre-Rating': round(team_b_pre_mu, 1),
            'Team A Post-Rating': round(team_a_post_mu, 1),
            'Team B Post-Rating': round(team_b_post_mu, 1),
            'Rating Change A': round(team_a_post_mu - team_a_pre_mu, 1),
            'Rating Change B': round(team_b_post_mu - team_b_pre_mu, 1)
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


def save_results(ratings_df, stats_df, history_df, session_df, data_dir=None):
    """
    Save all results to CSV files with appropriate timestamps.
    
    Args:
        ratings_df: DataFrame containing player TrueSkill ratings
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
    
    # Merge player stats with TrueSkill ratings
    combined_df = pd.merge(ratings_df, stats_df, on="Player")
    
    # Save combined results to player_trueskill_ratings.csv
    combined_path = os.path.join(save_dir, "player_trueskill_ratings.csv")
    combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
    
    # Generate timestamp for the match history file
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save match history with timestamp in filename
    match_history_filename = f"match_history_with_trueskill_{timestamp}.csv"
    history_path = os.path.join(save_dir, match_history_filename)
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    
    # Save session stats with timestamp in filename
    session_stats_filename = f"session_stats_{timestamp}.csv"
    session_path = os.path.join(save_dir, session_stats_filename)
    if not session_df.empty:
        session_df.to_csv(session_path, index=False, encoding="utf-8-sig")
    
    print(f"\nCombined TrueSkill ratings and statistics saved to '{combined_path}'")
    print(f"Match history with TrueSkill changes saved to '{history_path}'")
    print(f"Current session statistics saved to '{session_path}'")


def calculate_trueskill_ratings(excel_path, sheet_name="Schedule", data_dir=None):
    """
    Calculate TrueSkill ratings for players based on match results in the Excel file.
    Uses existing ratings and statistics from player_trueskill_ratings.csv if available.
    
    Args:
        excel_path: Path to the Excel file containing match results
        sheet_name: Name of the sheet to read from Excel file
        data_dir: Directory to save files to and look for existing data
        
    Returns:
        Tuple containing DataFrames for TrueSkill ratings, match history, player statistics, and session statistics
    """
    # Initialize session stats tracker
    session_stats = defaultdict(lambda: {
        "total": {"games": 0, "wins": 0},
        "male_double": {"games": 0, "wins": 0},
        "female_double": {"games": 0, "wins": 0},
        "mixed": {"games": 0, "wins": 0}
    })
    
    # Load existing player data
    player_ratings, player_stats = load_existing_player_data(data_dir)
    
    # Store initial ratings to calculate changes after the session
    initial_ratings = {player: Rating(mu=rating.mu, sigma=rating.sigma) 
                      for player, rating in player_ratings.items()}
    
    # Extract matches from Excel file
    matches = extract_matches_from_excel(excel_path, sheet_name)
    
    # Process matches and update TrueSkill ratings and statistics
    match_history = []
    for court, match, score, _ in matches:
        match_record = process_match(
            court, match, score,
            player_ratings, player_stats, session_stats
        )
        if match_record:
            match_history.append(match_record)
    
    # Convert TrueSkill dictionary to DataFrame for display
    ratings_data = []
    for player, rating in player_ratings.items():
        ratings_data.append({
            'Player': player,
            'Mean': rating.mu,
            'Sigma': rating.sigma,
            'Conservative': rating.mu - 3 * rating.sigma,  # Conservative estimate
            'Rating': rating.mu  # For compatibility with ELO display
        })
    
    ratings_df = pd.DataFrame(ratings_data)
    # Sort by conservative rating (mean - 3*sigma) for more stable ranking
    ratings_df = ratings_df.sort_values(by='Conservative', ascending=False).reset_index(drop=True)
    
    # Create DataFrames for match history and player statistics
    history_df = pd.DataFrame(match_history)
    stats_df = create_stats_dataframe(player_stats)
    session_df = create_stats_dataframe(session_stats)
    
    # Calculate rating changes for this session
    rating_changes = {}
    for player, final_rating in player_ratings.items():
        # If player is new (not in initial_ratings), use default mu
        initial_mu = initial_ratings.get(player, Rating(mu=ts_env.mu, sigma=ts_env.sigma)).mu
        rating_changes[player] = round(final_rating.mu - initial_mu, 1)
    
    # Add rating change column to session_df
    session_df['rating_change'] = session_df['Player'].map(rating_changes)
    
    # Print final TrueSkill ratings
    print("\n=== FINAL TRUESKILL RATINGS ===")
    display_df = ratings_df[['Player', 'Mean', 'Sigma', 'Conservative']]
    print(display_df.to_string(float_format='%.1f'))
    
    # Save all results to files
    save_results(ratings_df, stats_df, history_df, session_df, data_dir)
    
    return ratings_df, history_df, stats_df, session_df


if __name__ == "__main__":
    # Example usage:
    # Specify the path to your Excel file containing match results
    ratings_df, history_df, stats_df, session_df = calculate_trueskill_ratings('./results/20250604.xlsx')
