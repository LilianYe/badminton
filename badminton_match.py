from elo_system import load_existing_player_data
from scheduler import (generate_rotation, COURT_SIZE)
from validators import (check_play_times, check_partnerships, check_consecutive_rounds,
                        check_elo_balance, check_gender_balance,
                        check_opponent_frequency, save_schedule_to_excel)


def generate_schedule(players, court_count, start_hour, elo_threshold, game_per_player, team_elo_diff):
    """
    Generate a balanced badminton schedule
    
    Args:
        players: List of player names
        court_count: Number of courts available
        start_hour: Starting hour for the matches (24h format)
        elo_threshold: Maximum allowed ELO difference between teams
        game_per_player: Number of games each player should play
        team_elo_diff: Maximum allowed ELO difference between teammates
    """
    if len(players) * game_per_player % COURT_SIZE != 0:
        raise ValueError("The total number of games must be divisible by the number of players per court.")
    
    # Load player ELO ratings
    player_elos, _ = load_existing_player_data()
    
    # Generate rotation with gender-balanced courts
    rest_schedule, rounds_lineups = generate_rotation(
        players, court_count, game_per_player, elo_threshold=elo_threshold, 
        player_elos=player_elos, team_elo_diff=team_elo_diff)
    
    # Basic validity checks
    is_valid, play_counts = check_play_times(players, rounds_lineups, game_per_player=game_per_player)
    
    # Check partnerships
    partnerships_valid, repeated_pairs, _ = check_partnerships(rounds_lineups)
    print(f"Partnership check: {'✓ No repeated partnerships' if partnerships_valid else '✗ Has repeated partnerships'}")
    
    # Check gender balance
    gender_valid, imbalanced_courts = check_gender_balance(rounds_lineups)
    print(f"Gender balance check: {'✓ All courts gender-balanced' if gender_valid else '✗ Some courts are imbalanced'}")
    
    # Check ELO balance
    elo_valid, elo_imbalanced = check_elo_balance(rounds_lineups, player_elos, elo_threshold)
    print(f"ELO balance check: {'✓ All courts within ELO threshold' if elo_valid else '✗ Some courts exceed ELO threshold'}")
    
    # Check opponent frequency
    opponent_valid, frequent_matchups = check_opponent_frequency(rounds_lineups, max_opponent_frequency=game_per_player // 2)
    print(f"Opponent frequency check: {'✓ All matchups within frequency limit' if opponent_valid else '✗ Some matchups exceed limit'}")
    
    # Check consecutive rounds
    consecutive_valid, consecutive_streaks = check_consecutive_rounds(players, rounds_lineups)
    print(f"Consecutive rounds check: {'✓ No excessive consecutive play' if consecutive_valid else '✗ Some players have too many consecutive rounds'}")
    
    # Print any ELO imbalances if they exist
    if not elo_valid:
        print("Courts with ELO imbalance:")
        for round_num, court_idx, team1_elo, team2_elo, elo_diff in elo_imbalanced:
            print(f"  Round {round_num}, Court {court_idx}: Team ELOs {team1_elo:.1f} vs {team2_elo:.1f} (Diff: {elo_diff:.1f})")
    output_file = f"badminton_schedule_{elo_threshold}_{team_elo_diff}.xlsx"
    # Save the schedule to Excel
    save_schedule_to_excel(rest_schedule, rounds_lineups, output_file, start_hour=start_hour)
    return rounds_lineups 


if __name__ == "__main__":
    players = [ "敏敏子(F)", "cbt", "曹大", "Max", "Yunjie", "张晴川", 
               "🐟🍃", "Jing(F)", "ai(F)", "Damien", "MFive(F)", "乌拉乌拉", 
               "shuya(F)", "Yummy(F)", "廖俊杰", "尼古丁", "Louis", 
               "Acaprice", "方文", "米兰的小铁匠", "ian", "大米", "gdc", "Jensen", "OwenWei", "疏朗(F)"]
    # check_existing_schedule("badminton_schedule_20250614.xlsx")
    for i in range(100):
        try:
            rounds = generate_schedule(players, court_count=5, start_hour=17, elo_threshold=90, game_per_player=8, team_elo_diff=200)
            break
        except ValueError as e:
            print(f"Error generating schedule: {e}")
    
    # all_players = [
    #     "cbt", "Yunjie", "曹大", "张晴川", "ai(F)", "Jing(F)", "Damien", "Plastic", "Jieling(F)", 
    #     "dianhsu", "Max", "gdc", "MFive(F)", "🐟🍃", "卷卷(F)", "yy(F)", "乌拉乌拉", 
    #     "米兰的小铁匠", "敏敏子(F)", "尼古丁", "一顿饭", "疏朗(F)", "z", "杨昆",
    #  "星际宇航员", "李娜(F)", "颜若儒(F)", "simonBW", "安元植", "熊猫",  "liyu", "Chao",
    #  "destiny(F)", "李东勇",  'JianjunLv', "Yummy(F)", "王威", "Louis", "毛艺钧", 
    # "方文", "shuya(F)", "Acaprice", "廖俊杰", "ian", "大米",  "Jensen", "OwenWei"
    # ]
