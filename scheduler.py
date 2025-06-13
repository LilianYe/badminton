import random
import itertools
from collections import defaultdict
from elo_system import compute_team_elo

COURT_SIZE = 4  # Number of players per court (2 per team)
global_partnerships = defaultdict(int)  # Global partnerships tracker
global_opponents = defaultdict(int)  # Global opponents tracker


def generate_rotation(players, court_count, game_per_player, elo_threshold, player_elos, team_elo_diff):
    """
    Generate a rotation schedule with rest periods and gender-balanced courts.
    Automatically calculates the optimal number of rounds needed for each player to play same number of games.
    
    Args:
        players: List of player names
        court_count: Number of courts available (default=3)
        game_per_player: Number of games each player should play
        elo_threshold: Maximum allowed ELO difference between teams
        player_elos: Dictionary of player ELO ratings
    
    Returns:
        Tuple of (rest_schedule, rounds_lineups)
    """
    # Initialize global dictionaries for tracking partnerships and opponents
    global global_partnerships, global_opponents
    # Maximum number of times two players can play against each other
    max_opponent_frequency = game_per_player // 2
    
    female_players = [p for p in players if p.endswith("(F)")]
    female_set = set(female_players)
    total_players = len(players)
    
    # Calculate total player-games needed (each player plays game_per_player games)
    total_player_games = total_players * game_per_player
    
    # Calculate how many player-slots we have per round
    player_slots_per_round = court_count * COURT_SIZE
    
    # Calculate number of rounds needed (may have a fraction)
    rounds_float = total_player_games / player_slots_per_round
    full_rounds = int(rounds_float)  # Integer part
    
    # Determine if we need a partial round at the end
    has_partial_round = rounds_float > full_rounds
    rounds = full_rounds + (1 if has_partial_round else 0)
    
    # Calculate how many courts are needed in the last round
    last_round_courts = court_count
    if has_partial_round:
        remaining_player_games = total_player_games - (full_rounds * player_slots_per_round)
        last_round_courts = remaining_player_games // COURT_SIZE
        # Add one more court if there are leftover players
        if remaining_player_games % COURT_SIZE != 0:
            last_round_courts += 1
    
    print(f"Schedule parameters:")
    print(f"- Total players: {total_players}")
    print(f"- Courts per round: {court_count} (except possibly last round)")
    print(f"- Players per court: {COURT_SIZE}")
    print(f"- Target games per player: {game_per_player}")
    print(f"- Calculated rounds needed: {rounds} ({full_rounds} full rounds + {1 if has_partial_round else 0} partial round)")
    print(f"- Courts in last round: {last_round_courts}")
    print(f"- max_opponent_frequency: {max_opponent_frequency}")
    print(f"- female players size: {len(female_players)}")
    # Create a list of court counts for each round
    courts_per_round = [court_count] * (rounds - 1) + [last_round_courts]
    
    # Calculate rest players for each round based on court count
    rest_per_round = []
    for round_courts in courts_per_round:
        active_players = round_courts * COURT_SIZE
        rest_players = total_players - active_players
        rest_per_round.append(rest_players)
    
    # Initialize rest schedule with empty lists
    rest_schedule = [[] for _ in range(rounds)]
    
    # Track rest counts for each player
    rest_counts = {p: 0 for p in players}
    
    # Expected rest count for each player (total rounds - game_per_player games)
    target_rest_count = rounds - game_per_player
    
    # Initialize player consecutive activity tracking
    player_consecutive_active = {p: 0 for p in players}
    
    # Generate rest schedule using backtracking with variable rest counts per round
    print("Attempting to generate rest schedule using backtracking...")
    backtrack_rest_schedule_variable(
        players, rest_schedule, rest_counts, 0, rounds,
        rest_per_round, female_set, female_players, target_rest_count,
        player_consecutive_active  # Pass the consecutive rounds tracker
    )
    print("Rest schedule:", rest_schedule)
    
    # Courts lineup generation with backtracking
    rounds_lineups = []
    max_attempts = 500  # Try different rest schedules if needed
    for attempt in range(max_attempts):
        success = True
        temp_lineups = []
                
        for r in range(rounds):
            active = [p for p in players if p not in rest_schedule[r]]
            active_females = [p for p in active if p in female_set]
            active_males = [p for p in active if p not in female_set]
            
            # Use the correct number of courts for this round
            round_court_count = courts_per_round[r]
            # Try to generate a valid court assignment for this round
            courts = generate_courts_for_round(active_females, active_males, round_court_count, 
                                     female_set, player_elos, elo_threshold, max_opponent_frequency, team_elo_diff)
            
            if courts is None or len(courts) < round_court_count:
                # print(f"Failed to generate valid courts for round {r+1}, retrying with different arrangement")
                success = False
                # Clear any partnerships recorded during this failed attempt
                global_partnerships = defaultdict(int)  # Reset to empty
                global_opponents = defaultdict(int)
                break
                
            temp_lineups.append(courts)
            
        if success:
            rounds_lineups = temp_lineups
            break
            
        if attempt < max_attempts - 1:
            # Shuffle rest schedule for another attempt
            for r in range(rounds):
                random.shuffle(rest_schedule[r])
    
    if not rounds_lineups:
        raise ValueError("Failed to generate a valid schedule after multiple attempts")
        
    return rest_schedule, rounds_lineups


def backtrack_rest_schedule_variable(players, rest_schedule, rest_counts, current_round, total_rounds,
                                  rest_per_round, female_set, female_players, target_rest_count,
                                  player_consecutive_active):
    """
    Optimized recursive backtracking for rest schedule generation with consecutive rounds limit
    """
    # Base case: we've completed all rounds
    if current_round == total_rounds:
        return True
    
    # How many players need to rest this round
    rest_count_needed = rest_per_round[current_round]
    rounds_left = total_rounds - current_round
    
    # 1. IDENTIFY PLAYERS WHO MUST REST
    
    # Must rest due to consecutive active rounds (limit is 5)
    must_rest_consecutive = [p for p in players 
                           if player_consecutive_active.get(p, 0) >= 5]
    
    # Must rest due to remaining rest requirements
    rest_needed = {p: target_rest_count - rest_counts[p] for p in players}
    must_rest_count = [p for p, needed in rest_needed.items() 
                      if needed > 0 and needed > rounds_left - 1]
    
    # Combined must-rest list
    must_rest = list(set(must_rest_consecutive + must_rest_count))
    
    # 2. VALIDATE BASIC CONSTRAINTS
    
    # Can't satisfy if too many players must rest
    if len(must_rest) > rest_count_needed:
        return False
    
    # Can't satisfy if not enough players still need rest
    if sum(1 for p in players if rest_counts[p] < target_rest_count) < rest_count_needed:
        return False
    
    # 3. IDENTIFY CANDIDATES FOR RESTING
    
    # Remaining valid candidates (need rest but not in must-rest)
    valid_candidates = [p for p in players 
                        if rest_counts[p] < target_rest_count and p not in must_rest]
    
    # Split by gender
    female_candidates = [p for p in valid_candidates if p in female_set]
    male_candidates = [p for p in valid_candidates if p not in female_set]
    must_rest_females = [p for p in must_rest if p in female_set]
    
    # 4. GENERATE REST COMBINATIONS CONSIDERING GENDER BALANCE
    
    combinations_to_try = []
    rest_slots_remaining = rest_count_needed - len(must_rest)
    
    # Need to maintain even number of active females for court balance
    current_female_count = len(must_rest_females)
    active_females = len(female_players) - current_female_count
    needed_females_to_rest = active_females % 2  # 1 if odd active, 0 if even
    
    # Priority sorting - players with more consecutive rounds and more rest needed
    female_priority = sorted(
        female_candidates, 
        key=lambda p: (player_consecutive_active.get(p, 0), rest_needed[p]), 
        reverse=True
    )
    male_priority = sorted(
        male_candidates,
        key=lambda p: (player_consecutive_active.get(p, 0), rest_needed[p]),
        reverse=True
    )
    
    # Limit search space
    max_female_samples = min(len(female_priority), 8)
    max_male_samples = min(len(male_priority), 8)
    female_priority = female_priority[:max_female_samples]
    male_priority = male_priority[:max_male_samples]
    
    # Generate appropriate gender-balanced combinations
    if needed_females_to_rest == 1:  # Need odd females resting
        for offset in range(min(3, (len(female_priority) + 1) // 2)):
            females_to_add = 1 + (offset * 2)  # 1, 3, 5...
            if females_to_add > rest_slots_remaining:
                continue
                
            males_to_add = rest_slots_remaining - females_to_add
            if males_to_add > len(male_priority):
                continue
                
            # Try combinations with this gender distribution
            for female_combo in itertools.combinations(female_priority, min(females_to_add, len(female_priority))):
                for male_combo in itertools.combinations(male_priority, min(males_to_add, len(male_priority))):
                    # Check if we already have enough combinations 
                    if len(combinations_to_try) >= 10:
                        break
                    combinations_to_try.append(list(must_rest) + list(female_combo) + list(male_combo))
    else:  # Need even females resting
        for offset in range(min(3, len(female_priority) // 2 + 1)):
            females_to_add = offset * 2  # 0, 2, 4...
            if females_to_add > rest_slots_remaining:
                continue
                
            males_to_add = rest_slots_remaining - females_to_add
            if males_to_add > len(male_priority):
                continue
                
            # Try combinations with this gender distribution
            for female_combo in itertools.combinations(female_priority, min(females_to_add, len(female_priority))):
                for male_combo in itertools.combinations(male_priority, min(males_to_add, len(male_priority))):
                    # Check if we already have enough combinations 
                    if len(combinations_to_try) >= 10:
                        break
                    combinations_to_try.append(list(must_rest) + list(female_combo) + list(male_combo))
    
    # Add variety
    if combinations_to_try:
        random.shuffle(combinations_to_try)
    
    # 5. TRY EACH COMBINATION WITH BACKTRACKING
    
    for combo in combinations_to_try:
        # Create a copy of consecutive active rounds
        player_consecutive_active_copy = player_consecutive_active.copy()
        
        # Add players to rest schedule
        for player in combo:
            rest_schedule[current_round].append(player)
            rest_counts[player] += 1
            player_consecutive_active_copy[player] = 0  # Reset consecutive count for resting players
        
        # Update consecutive active rounds for players not in combo
        for player in players:
            if player not in combo:  # Player is active this round
                player_consecutive_active_copy[player] = player_consecutive_active_copy.get(player, 0) + 1
        
        # Recursively try to complete the schedule
        if backtrack_rest_schedule_variable(
                players, rest_schedule, rest_counts, 
                current_round + 1, total_rounds, 
                rest_per_round, female_set, female_players, 
                target_rest_count, player_consecutive_active_copy
            ):
            return True
        
        # If we get here, this combination didn't work - backtrack
        for player in combo:
            rest_schedule[current_round].remove(player)
            rest_counts[player] -= 1
    
    # If we tried all combinations and none worked, return False
    return False


def generate_courts_for_round(females, males, court_count, female_set, player_elos, 
                            elo_threshold, max_opponent_frequency, team_elo_diff):
    """
    Generate court assignments for a single round using backtracking
    
    Args:
        females: List of active female players
        males: List of active male players
        court_count: Number of courts to fill
        female_set: Set of female players for quick lookup
        player_elos: Dictionary of player ELO ratings
        elo_threshold: Maximum allowed ELO difference between teams
        max_opponent_frequency: Maximum allowed times two players can face each other
        team_elo_diff: Maximum allowed ELO difference between teammates
        
    Returns:
        List of courts if successful, None otherwise
    """
    courts = []
    players_used = set()
    
    # Make copies of our lists to work with
    females = females.copy()
    males = males.copy()
    
    # Try different distributions using backtracking
    if not backtrack_courts(courts, females, males, court_count, female_set, players_used, 
                          player_elos, elo_threshold, max_opponent_frequency, team_elo_diff):
        return None
        
    return courts


def check_team_elo(team1, team2, player_elos, elo_threshold):
    elo_team1 = compute_team_elo(player_elos[team1[0]], player_elos[team1[1]])
    elo_team2 = compute_team_elo(player_elos[team2[0]], player_elos[team2[1]])
    return abs(elo_team1 - elo_team2) <= elo_threshold


def check_opponents_valid(team1, team2, max_opponent_frequency):
    """Check if placing these teams against each other would exceed opponent frequency limits"""
    global global_opponents
    
    for p1 in team1:
        for p2 in team2:
            opponent_pair = tuple(sorted([p1, p2]))
            # Check if these players have faced each other too many times already
            if global_opponents.get(opponent_pair, 0) >= max_opponent_frequency:
                return False
    return True


def backtrack_courts(courts, females, males, court_count, female_set, used_players, 
                    player_elos, elo_threshold, max_opponent_frequency, team_elo_diff):
    """
    Recursively build court assignments using backtracking
    
    Args:
        courts: Current list of courts being built
        females: Available female players
        males: Available male players
        court_count: Total number of courts needed
        female_set: Set of female players for quick lookup
        used_players: Set of players already assigned in this round
        player_elos: Dictionary of player ELO ratings
        elo_threshold: Maximum allowed ELO difference between teams
        max_opponent_frequency: Maximum number of times two players can face each other
        team_elo_diff: Maximum allowed ELO difference between teammates
        
    Returns:
        True if successful, False otherwise
    """
    global global_partnerships, global_opponents
    
    # Base case: we've filled all courts
    if len(courts) == court_count:
        return True
    
    # Try different gender distributions for this court
    court_options = []
    
    # Try 2 females + 2 males (if available)
    if len(females) >= 2 and len(males) >= 2:
        court_options.append('mixed')
        
    # Try 4 females (if available)
    if len(females) >= 4:
        court_options.append('all_female')
        
    # Try 4 males (if available)
    if len(males) >= 4:
        court_options.append('all_male')
        
    # Shuffle options for variety
    random.shuffle(court_options)
    
    # Try each option
    for option in court_options:
        # Create a copy of our available players to restore if this path fails
        females_copy = females.copy()
        males_copy = males.copy()
        partnerships_snapshot = {k: v for k, v in global_partnerships.items()}
        opponents_snapshot = {k: v for k, v in global_opponents.items()}
        used_copy = used_players.copy()
        
        court = []
        success = False
        
        if option == 'mixed':
            # Try to form teams with 1 female + 1 male per team
            team1, team2 = [], []
            
            # Try to form team 1 (1F + 1M)
            for f1 in females_copy:
                if f1 in used_copy:
                    continue
                    
                for m1 in males_copy:
                    if m1 in used_copy:
                        continue
                    
                    # Check global partnerships
                    pair = tuple(sorted([f1, m1]))
                    if pair in global_partnerships:
                        continue
                        
                    # Check teammate ELO compatibility
                    if not check_teammate_elo_compatibility(f1, m1, player_elos, team_elo_diff):
                        continue
                        
                    team1 = [f1, m1]
                    global_partnerships[pair] = global_partnerships.get(pair, 0) + 1
                    used_copy.add(f1)
                    used_copy.add(m1)
                    break
                    
                if team1:
                    break
            
            # If we couldn't form team1, skip this option
            if not team1:
                continue
                
            # Try to form team 2 (1F + 1M)
            for f2 in females_copy:
                if f2 in used_copy:
                    continue
                    
                for m2 in males_copy:
                    if m2 in used_copy:
                        continue
                        
                    # Check global partnerships
                    pair = tuple(sorted([f2, m2]))
                    if pair in global_partnerships:
                        continue 
                    # Skip if ELO difference is too large
                    if not check_team_elo(team1, [f2, m2], player_elos, elo_threshold):
                        continue
                    # Check opponent frequency
                    if not check_opponents_valid(team1, [f2, m2], max_opponent_frequency):
                        continue
                    if not check_teammate_elo_compatibility(f2, m2, player_elos, team_elo_diff):
                        continue
                    team2 = [f2, m2]
                    global_partnerships[pair] = global_partnerships.get(pair, 0) + 1
                    for p1 in team1:
                        for p2 in team2:
                            opponent_pair = tuple(sorted([p1, p2]))
                            global_opponents[opponent_pair] = global_opponents.get(opponent_pair, 0) + 1
                    used_copy.add(f2)
                    used_copy.add(m2)
                    break
                    
                if team2:
                    break
                    
            # If we found both teams, add them to the court
            if team1 and team2:
                court = team1 + team2
                females = [f for f in females_copy if f not in used_copy]
                males = [m for m in males_copy if m not in used_copy]
                success = True
            
        elif option == 'all_female':
            # Try to form 2 teams with 2 females each
            team1, team2 = [], []
            female_pairs = list(itertools.combinations(females_copy, 2))
            random.shuffle(female_pairs)
            
            # Try to form team 1 (2F)
            for f1, f2 in female_pairs:
                if f1 in used_copy or f2 in used_copy:
                    continue
                    
                # Check global partnerships
                pair = tuple(sorted([f1, f2]))
                if pair in global_partnerships:
                    continue
                if not check_teammate_elo_compatibility(f1, f2, player_elos, team_elo_diff):
                    continue
                team1 = [f1, f2]
                global_partnerships[pair] = global_partnerships.get(pair, 0) + 1
                used_copy.add(f1)
                used_copy.add(f2)
                break
                
            # If we couldn't form team1, skip this option
            if not team1:
                continue
                
            # Try to form team 2 (2F)
            female_pairs = list(itertools.combinations([f for f in females_copy if f not in used_copy], 2))
            random.shuffle(female_pairs)
            
            for f1, f2 in female_pairs:
                # Check global partnerships
                pair = tuple(sorted([f1, f2]))
                if pair in global_partnerships:
                    continue   
                # Skip if ELO difference is too large
                if not check_team_elo(team1, [f1, f2], player_elos, elo_threshold):
                    continue
                if not check_opponents_valid(team1, [f1, f2], max_opponent_frequency):
                    continue
                if not check_teammate_elo_compatibility(f1, f2, player_elos, team_elo_diff):
                    continue
                team2 = [f1, f2]
                global_partnerships[pair] = global_partnerships.get(pair, 0) + 1
                for p1 in team1:
                    for p2 in team2:
                        opponent_pair = tuple(sorted([p1, p2]))
                        global_opponents[opponent_pair] = global_opponents.get(opponent_pair, 0) + 1
                used_copy.add(f1)
                used_copy.add(f2)
                break
                
            # If we found both teams, add them to the court
            if team1 and team2:
                court = team1 + team2
                females = [f for f in females_copy if f not in used_copy]
                success = True
                
        elif option == 'all_male':
            # Try to form 2 teams with 2 males each
            team1, team2 = [], []
            male_pairs = list(itertools.combinations(males_copy, 2))
            random.shuffle(male_pairs)
            
            # Try to form team 1 (2M)
            for m1, m2 in male_pairs:
                if m1 in used_copy or m2 in used_copy:
                    continue
                    
                # Check global partnerships
                pair = tuple(sorted([m1, m2]))
                if pair in global_partnerships:
                    continue
                if not check_teammate_elo_compatibility(m1, m2, player_elos, team_elo_diff):
                    continue
                team1 = [m1, m2]
                global_partnerships[pair] = global_partnerships.get(pair, 0) + 1
                used_copy.add(m1)
                used_copy.add(m2)
                break
                
            # If we couldn't form team1, skip this option
            if not team1:
                continue
                
            # Try to form team 2 (2M)
            male_pairs = list(itertools.combinations([m for m in males_copy if m not in used_copy], 2))
            random.shuffle(male_pairs)
            
            for m1, m2 in male_pairs:
                # Check global partnerships
                pair = tuple(sorted([m1, m2]))
                if pair in global_partnerships:
                    continue
                # Skip if ELO difference is too large
                if not check_team_elo(team1, [m1, m2], player_elos, elo_threshold):
                    continue
                if not check_opponents_valid(team1, [m1, m2], max_opponent_frequency):
                    continue
                if not check_teammate_elo_compatibility(m1, m2, player_elos, team_elo_diff):
                    continue
                team2 = [m1, m2]
                global_partnerships[pair] = global_partnerships.get(pair, 0) + 1
                for p1 in team1:
                    for p2 in team2:
                        opponent_pair = tuple(sorted([p1, p2]))
                        global_opponents[opponent_pair] = global_opponents.get(opponent_pair, 0) + 1
                used_copy.add(m1)
                used_copy.add(m2)
                break
                
            # If we found both teams, add them to the court
            if team1 and team2:
                court = team1 + team2
                males = [m for m in males_copy if m not in used_copy]
                success = True
                
        # If we successfully created a court
        if success and court:
            courts.append(court)
            used_players = used_copy
            
            # Continue with the next court
            if backtrack_courts(courts, females, males, court_count, female_set, used_players, player_elos, elo_threshold, max_opponent_frequency, team_elo_diff):
                return True
                
            # If we can't complete the schedule with this court, backtrack
            courts.pop()
            # Restore partnerships snapshot on backtracking
            global_partnerships = partnerships_snapshot
            global_opponents = opponents_snapshot
            
    # If we tried all options and none worked, backtrack
    return False


def check_teammate_elo_compatibility(player1, player2, player_elos, team_elo_diff):
    """
    Check if two players have compatible ELO ratings to be teammates.
    
    Args:
        player1: First player name
        player2: Second player name
        player_elos: Dictionary of player ELO ratings
        team_elo_diff: Maximum allowed ELO difference between teammates
        
    Returns:
        Boolean indicating if players can be teammates based on ELO difference
    """
    return abs(player_elos[player1] - player_elos[player2]) <= team_elo_diff
