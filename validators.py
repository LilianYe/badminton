from collections import defaultdict
import scheduler
import pandas as pd
import os
import random
import datetime
from elo_system import load_existing_player_data, compute_team_elo


def check_play_times(players, rounds_lineups, game_per_player):
    """
    Check that each player plays exactly game_per_player times
    
    Args:
        players: List of player names
        rounds_lineups: List of round lineups, each containing court assignments
        game_per_player: Number of games each player should play
        
    Returns:
        tuple: (is_valid, play_counts) - Boolean indicating if all players play exactly game_per_player times,
               and a dictionary with each player's play count
    """
    # Initialize play counters
    play_counts = {p: 0 for p in players}
    
    # Count play times for each player
    for round_courts in rounds_lineups:
        for court in round_courts:
            for player in court:
                play_counts[player] += 1
    
    # Check if all players play exactly game_per_player times
    is_valid = all(count == game_per_player for count in play_counts.values())
    
    # Find any players with incorrect play counts
    if not is_valid:
        incorrect_players = {p: c for p, c in play_counts.items() if c != game_per_player}
        print(f"Warning: The following players don't play exactly {game_per_player} times:")
        for player, count in incorrect_players.items():
            print(f"  {player}: {count} times")
    else:
        print(f"✓ All players play exactly {game_per_player} times as expected.")        


def check_partnerships(rounds_lineups):
    """
    Check if any two players are scheduled to play as partners more than once.
    
    Args:
        rounds_lineups: List of round lineups, each containing court assignments
        
    Returns:
        tuple: (is_valid, repeated_pairs, partnership_counts) - Boolean indicating if all partnerships are unique,
               a list of repeated partnerships, and a dictionary counting each partnership
    """
    # Track all partnerships that have occurred
    partnerships = defaultdict(int)
    
    for round_courts in rounds_lineups:
        for court in round_courts:
            # Each court has 4 players, forming 2 teams
            team1 = (court[0], court[1])
            team2 = (court[2], court[3])
            
            # Check if these partnerships already exist
            for team in [team1, team2]:
                # Sort the team to ensure consistent ordering regardless of player order
                sorted_team = tuple(sorted(team))
                partnerships[sorted_team] += 1
    
    # Find repeated partnerships
    repeated_pairs = [pair for pair, count in partnerships.items() if count > 1]
    for pair, count in partnerships.items():
        if count > 1:
            print(f"Partnership {pair} has been scheduled {count} times.")
    is_valid = len(repeated_pairs) == 0
    if is_valid:
        print("✓ All partnerships are unique.")


def check_gender_balance(rounds_lineups):
    """
    Check if all courts have gender-balanced teams (same number of females on each team)
    
    Args:
        rounds_lineups: List of round lineups, each containing court assignments
        
    Returns:
        tuple: (is_valid, imbalanced_courts) - Boolean indicating if all courts are gender-balanced,
               and a list of imbalanced courts
    """
    imbalanced_courts = []
    
    for round_num, round_courts in enumerate(rounds_lineups, 1):
        for court_idx, court in enumerate(round_courts, 1):
            # Count females in team 1
            team1_females = sum(1 for player in court[0:2] if player.endswith("(F)"))
            # Count females in team 2
            team2_females = sum(1 for player in court[2:4] if player.endswith("(F)"))

            if team1_females != team2_females:
                imbalanced_courts.append((round_num, court_idx, team1_females, team2_females))
    
    is_valid = len(imbalanced_courts) == 0
    if is_valid:
        print("✓ All courts ars gender-balanced.")
    else:
        print("The following courts are not gender-balanced:")
        for round_num, court_idx, team1_females, team2_females in imbalanced_courts:
            print(f"  Round {round_num}, Court {court_idx}: Team 1 has {team1_females} females, Team 2 has {team2_females} females")
    return is_valid, imbalanced_courts


def check_elo_balance(rounds_lineups, player_elos, elo_threshold):
    """
    Check if all courts have ELO-balanced teams (difference in team ELO <= threshold)
    
    Args:
        rounds_lineups: List of round lineups, each containing court assignments
        player_elos: Dictionary of player ELO ratings
        elo_threshold: Maximum allowed ELO difference between teams
        
    Returns:
        tuple: (is_valid, imbalanced_courts) - Boolean indicating if all courts are ELO-balanced,
               and a list of imbalanced courts with their ELO differences
    """
    imbalanced_courts = []
    
    for round_num, round_courts in enumerate(rounds_lineups, 1):
        for court_idx, court in enumerate(round_courts, 1):
            team1_elo = compute_team_elo(player_elos[court[0]], player_elos[court[1]])
            team2_elo = compute_team_elo(player_elos[court[2]], player_elos[court[3]])
            elo_diff = abs(team1_elo - team2_elo)
            if elo_diff > elo_threshold:
                imbalanced_courts.append((round_num, court_idx, team1_elo, team2_elo, elo_diff))
    
    is_valid = len(imbalanced_courts) == 0
    if is_valid:
        print("✓ All courts are ELO-balanced.")
    else:
        print("The following courts have ELO imbalances:")
        for round_num, court_idx, team1_elo, team2_elo, elo_diff in imbalanced_courts:
            print(f"  Round {round_num}, Court {court_idx}: Team ELOs {team1_elo:.1f} vs {team2_elo:.1f} (Diff: {elo_diff:.1f})")


def check_opponent_frequency(rounds_lineups, max_opponent_frequency):
    """
    Check if any two players are scheduled to play against each other more than the allowed maximum.
    
    Args:
        rounds_lineups: List of round lineups, each containing court assignments
        max_opponent_frequency: Maximum number of times two players can face each other
        
    Returns:
        tuple: (is_valid, frequent_matchups) - Boolean indicating if opponent frequency limits are met,
               and a list of matchups that exceed the limit
    """
    # Track all opponent encounters
    opponent_counts = defaultdict(int)
    
    for round_courts in rounds_lineups:
        for court in round_courts:
            # Each court has 4 players, forming 2 teams
            team1 = court[0:2]
            team2 = court[2:4]
            
            # Record opponent encounters
            for p1 in team1:
                for p2 in team2:
                    opponent_pair = tuple(sorted([p1, p2]))
                    opponent_counts[opponent_pair] += 1
    
    # Find frequent matchups
    frequent_matchups = [(pair, count) for pair, count in opponent_counts.items() 
                         if count > max_opponent_frequency]
    is_valid = len(frequent_matchups) == 0
    
    if not is_valid:
        print("The following players face each other too frequently:")
        for pair, count in sorted(frequent_matchups, key=lambda x: x[1], reverse=True):
            print(f"  {pair[0]} vs {pair[1]}: {count} times (limit: {max_opponent_frequency})")
    else:
        print(f"✓ No players face each other more than {max_opponent_frequency} times.")


def check_consecutive_rounds(players, rounds_lineups, max_consecutive_allowed):
    """
    Check if any player is scheduled to play in 4 or more consecutive rounds.
    
    Args:
        players: List of player names
        rounds_lineups: List of round lineups, each containing court assignments
        
    Returns:
        tuple: (is_valid, consecutive_players) - Boolean indicating if all players have sufficient rest,
               and a dictionary of players with their longest consecutive rounds streak
    """
    # Initialize tracking of consecutive rounds
    player_activity = {p: [] for p in players}
    all_players_in_schedule = set()
    # For each round, record if a player is active (1) or resting (0)
    for round_idx, round_courts in enumerate(rounds_lineups):
        # Get all active players in this round
        active_players = set()
        for court in round_courts:
            for player in court:
                active_players.add(player)
                all_players_in_schedule.add(player)
        # Record activity for each player
        for player in players:
            player_activity[player].append(1 if player in active_players else 0)

    # Check for consecutive rounds played
    max_consecutive = {}
    for player, activity in player_activity.items():
        # Split activity into streaks of consecutive 1s
        activity_str = ''.join(map(str, activity))
        streaks = [len(streak) for streak in activity_str.split('0') if streak]
        max_streak = max(streaks) if streaks else 0
        max_consecutive[player] = max_streak
    # Identify players with too many consecutive rounds
    problematic_players = {p: streak for p, streak in max_consecutive.items() if streak > max_consecutive_allowed}
    is_valid = len(problematic_players) == 0
    
    if not is_valid:
        print("The following players have too many consecutive rounds:")
        for player, streak in sorted(problematic_players.items(), key=lambda x: x[1], reverse=True):
            print(f"  {player}: {streak} consecutive rounds")
    else:
        print(f"✓ No players have too many consecutive rounds (max allowed: {max_consecutive_allowed})")


def save_schedule_to_excel(rest_schedule, rounds_lineups, output_file, start_hour=17):
    """
    Save the badminton schedule to an Excel file with all courts on a single line
    and referees on a separate line with support for variable number of courts per round
    """
    # Create a new Excel writer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    
    # Create data for schedule sheet
    schedule_data = []
    start_time = datetime.time(start_hour, 0)
    
    for round_num, (rest, courts) in enumerate(zip(rest_schedule, rounds_lineups), 1):
        # Calculate time range for this round
        round_start = datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(minutes=(round_num-1)*12)
        round_end = round_start + datetime.timedelta(minutes=12)
        time_range = f"{round_start.strftime('%H:%M')} - {round_end.strftime('%H:%M')}"
        
        # Assign referees (resting players) to courts
        referees = rest.copy()
        random.shuffle(referees)  # Randomize referee assignments
        
        # Get the number of courts for this round
        court_count = len(courts)
        
        # Create a row for round header and player matchups
        row = {"Round": f"Round {round_num} ({time_range})"}
        
        # Add all courts to the same row (without referee info)
        for court_num, court in enumerate(courts, 1):
            team1 = f"{court[0]}/{court[1]}"
            team2 = f"{court[2]}/{court[3]}"
            row[f"Court {court_num}"] = f"{team1} vs {team2}"
        
        schedule_data.append(row)
        
        # Add referee information as a separate row
        ref_row = {"Round": "Umpires"}
        for court_num in range(1, court_count+1):
            referee = referees[court_num-1] if court_num <= len(referees) else "N/A"
            ref_row[f"Court {court_num}"] = referee
        
        schedule_data.append(ref_row)
        
        # Add an empty row between rounds for better readability
        empty_row = {"Round": ""}
        for court_num in range(1, court_count+1):
            empty_row[f"Court {court_num}"] = ""
        schedule_data.append(empty_row)
    
    # Convert to DataFrame
    schedule_df = pd.DataFrame(schedule_data)
    
    # Write only the schedule DataFrame to Excel
    schedule_df.to_excel(writer, sheet_name="Schedule", index=False)
    
    # Auto-adjust column width
    worksheet = writer.sheets["Schedule"]
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if cell.value and len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[column_letter].width = adjusted_width
    
    # Save the Excel file
    writer.close()
    print(f"Schedule saved to {os.path.abspath(output_file)}")


def extract_rounds_from_excel(excel_path, sheet_name="Schedule"):
    """
    Extract match data from Excel file and format it as rounds_lineups.
    
    Args:
        excel_path: Path to Excel file containing match schedule
        sheet_name: Name of the sheet to read from Excel file
        
    Returns:
        List structure matching rounds_lineups format for validation
    """
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # Find rows that contain round headers
    round_header_rows = []
    for row in range(df.shape[0]):
        cell = str(df.iat[row, 0]) if not pd.isna(df.iat[row, 0]) else ""
        if "Round" in cell and any(str(i) in cell for i in range(1, 100)):
            round_header_rows.append(row)
    
    print(f"Found {len(round_header_rows)} round header rows")
    
    # Process each round
    rounds_lineups = []
    for i, row in enumerate(round_header_rows):
        current_round = []
        
        # Process match data in this row
        for col in range(1, df.shape[1]):
            cell = str(df.iat[row, col]) if not pd.isna(df.iat[row, col]) else ""
            if " vs " in cell:
                teams = cell.split(" vs ")
                if len(teams) == 2:
                    team1 = teams[0].strip().split("/")
                    team2 = teams[1].strip().split("/")
                    if len(team1) == 2 and len(team2) == 2:
                        court = [team1[0].strip(), team1[1].strip(), team2[0].strip(), team2[1].strip()]
                        current_round.append(court)
        
        # Add this round if it has courts
        if current_round:
            rounds_lineups.append(current_round)
    
    print(f"Extracted {len(rounds_lineups)} rounds with {sum(len(courts) for courts in rounds_lineups)} courts")
    return rounds_lineups


def check_existing_schedule(excel_path):
    """
    Check if the existing schedule in the Excel file matches the expected format.
    
    Args:
        excel_path: Path to Excel file containing match schedule
        sheet_name: Name of the sheet to read from Excel file
        
    Returns:
        Boolean indicating if the existing schedule is valid
    """
    # Load schedule from Excel
    rounds_lineups = extract_rounds_from_excel(excel_path)
    player_elos, _ = load_existing_player_data()
    elo_threshold = 70  # Example threshold, adjust as needed
    max_opponent_frequency = 3  # Example limit, adjust as needed
    # Get all players
    all_players = set()
    for round_courts in rounds_lineups:
        for court in round_courts:
            for player in court:
                all_players.add(player)
    all_players = list(all_players)
    # Run validation checks
    check_play_times(all_players, rounds_lineups, game_per_player=6)
    check_partnerships(rounds_lineups)
    check_gender_balance(rounds_lineups)
    check_elo_balance(rounds_lineups, player_elos, elo_threshold)
    check_opponent_frequency(rounds_lineups, max_opponent_frequency)
    check_consecutive_rounds(all_players, rounds_lineups, 3)
    expected_win_valid, _, _ = check_expected_win_balance(rounds_lineups, player_elos)


def check_expected_win_balance(rounds_lineups, player_elos):
    """
    Check if players have a balanced number of matches where their team has an ELO advantage.
    
    Args:
        rounds_lineups: List of round lineups, each containing court assignments
        player_elos: Dictionary of player ELO ratings
        
    Returns:
        tuple: (is_valid, player_expected_wins, player_expected_win_pct) - 
               Boolean indicating if expected win distribution is fair,
               dictionaries of each player's expected win count and percentages
    """
    # Initialize counters for each player
    players = set()
    for round_courts in rounds_lineups:
        for court in round_courts:
            for player in court:
                players.add(player)
    
    player_matches = {p: 0 for p in players}
    player_expected_wins = {p: 0 for p in players}
    
    # Analyze each match
    for round_courts in rounds_lineups:
        for court in round_courts:
            # Each court has 4 players: [team1_player1, team1_player2, team2_player1, team2_player2]
            team1 = court[0:2]
            team2 = court[2:4]
            
            # Calculate team average ELOs
            team1_avg_elo = (player_elos.get(team1[0]) + player_elos.get(team1[1])) / 2
            team2_avg_elo = (player_elos.get(team2[0]) + player_elos.get(team2[1])) / 2
            
            # For each player in team 1, check if their team has an advantage
            for player in team1:
                player_matches[player] += 1
                if team1_avg_elo > team2_avg_elo:
                    player_expected_wins[player] += 1
            
            # For each player in team 2, check if their team has an advantage
            for player in team2:
                player_matches[player] += 1
                if team2_avg_elo > team1_avg_elo:
                    player_expected_wins[player] += 1
    
    # Calculate expected win percentage for each player
    player_expected_win_pct = {}
    for player in players:
        if player_matches[player] > 0:
            player_expected_win_pct[player] = (player_expected_wins[player] / player_matches[player]) * 100
        else:
            player_expected_win_pct[player] = 0
    
    # Find the range of expected win percentages
    min_pct = min(player_expected_win_pct.values()) if player_expected_win_pct else 0
    max_pct = max(player_expected_win_pct.values()) if player_expected_win_pct else 0
    avg_pct = sum(player_expected_win_pct.values()) / len(player_expected_win_pct) if player_expected_win_pct else 0
    range_pct = max_pct - min_pct
    
    # Determine if the range is acceptable (e.g., within 25% points)
    threshold = 25  # Adjust this threshold as needed
    is_valid = range_pct <= threshold
    
    # Print results
    print(f"\nExpected Win Balance Check:")
    print(f"Expected win percentage range: {min_pct:.1f}% - {max_pct:.1f}% (range: {range_pct:.1f}%)")
    print(f"Average expected win percentage: {avg_pct:.1f}%")
    
    # Find players with extreme values
    low_players = {p: pct for p, pct in player_expected_win_pct.items() if pct < avg_pct - (threshold/2)}
    high_players = {p: pct for p, pct in player_expected_win_pct.items() if pct > avg_pct + (threshold/2)}
    
    if not is_valid:
        print("\nThe following players have significantly different expected win opportunities:")
        print("\nPlayers with LOW expected win opportunities:")
        for player, pct in sorted(low_players.items(), key=lambda x: x[1]):
            wins = player_expected_wins[player]
            matches = player_matches[player]
            print(f"  {player}: {pct:.1f}% ({wins}/{matches} matches with advantage)")
            
        print("\nPlayers with HIGH expected win opportunities:")
        for player, pct in sorted(high_players.items(), key=lambda x: x[1], reverse=True):
            wins = player_expected_wins[player]
            matches = player_matches[player]
            print(f"  {player}: {pct:.1f}% ({wins}/{matches} matches with advantage)")
    else:
        print(f"✓ All players have balanced expected win opportunities (range within {threshold}% points)")
    
    return is_valid, player_expected_wins, player_expected_win_pct

if __name__ == "__main__":
    # Example usage
    # Generate a schedule (uncomment to generate a new schedule)
    # players = ["Alice(F)", "Bob(M)", "Charlie(M)", "Diana(F)", "Eve(F)", "Frank(M)"]
    # rounds_lineups = scheduler.generate_schedule(players, court_count=2, start_hour=17, elo_threshold=70, game_per_player=3, team_elo_diff=50)
    
    # Check an existing schedule
    check_existing_schedule('badminton_schedule_70_300.xlsx')
