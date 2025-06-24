from elo_system import load_existing_player_data
from scheduler import (generate_rotation, COURT_SIZE)
from validators import (check_play_times, check_partnerships, check_consecutive_rounds,
                        check_elo_balance, check_gender_balance,
                        check_opponent_frequency, save_schedule_to_excel)


def generate_schedule(players, court_count, start_hour, elo_threshold, game_per_player, team_elo_diff, max_opponent_frequency, min_expected_wins):
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
        players, court_count, game_per_player, elo_threshold, 
        player_elos, team_elo_diff, max_opponent_frequency, min_expected_wins)
    
    # Basic validity checks
    check_play_times(players, rounds_lineups, game_per_player=game_per_player)
    check_partnerships(rounds_lineups)
    check_gender_balance(rounds_lineups)
    check_elo_balance(rounds_lineups, player_elos, elo_threshold)
    check_opponent_frequency(rounds_lineups, max_opponent_frequency)
    check_consecutive_rounds(players, rounds_lineups, 4)
    
    # Calculate average ELO difference
    avg_elo_diff = calculate_average_elo_difference(rounds_lineups, player_elos)
    print(f"Average ELO difference between teams: {avg_elo_diff:.2f}")
    
    # Update output file name to include average ELO difference
    output_file = f"badminton_schedule_{court_count}_{elo_threshold}_{team_elo_diff}_{max_opponent_frequency}_{avg_elo_diff:.0f}.xlsx"
    
    # Save the schedule to Excel
    save_schedule_to_excel(rest_schedule, rounds_lineups, output_file, start_hour=start_hour)
    return rounds_lineups 


def calculate_average_elo_difference(rounds_lineups, player_elos):
    """
    Calculate the average ELO difference between teams across all matches
    
    Args:
        rounds_lineups: List of round lineups, each containing court assignments
        player_elos: Dictionary of player ELO ratings
        
    Returns:
        float: Average ELO difference between teams
    """
    total_difference = 0
    match_count = 0
    
    for round_courts in rounds_lineups:
        for court in round_courts:
            # Each court has 4 players: [team1_player1, team1_player2, team2_player1, team2_player2]
            team1 = court[0:2]
            team2 = court[2:4]
            
            # Calculate team average ELOs
            team1_avg_elo = (player_elos.get(team1[0], 1500) + player_elos.get(team1[1], 1500)) / 2
            team2_avg_elo = (player_elos.get(team2[0], 1500) + player_elos.get(team2[1], 1500)) / 2
            
            # Calculate the absolute difference
            difference = abs(team1_avg_elo - team2_avg_elo)
            
            total_difference += difference
            match_count += 1
    
    # Calculate average
    if match_count > 0:
        return total_difference / match_count
    else:
        return 0


if __name__ == "__main__":
    players = [ "敏敏子(F)", 'liyu', "米兰的小铁匠", "simonBW", "熊猫",  "安元植", "颜若儒(F)", "毛艺钧",
               "李娜(F)", "destiny(F)", "伟帆", "浩南", "王威", "Chao", "一顿饭", "风", 
               "吳祎麟", "方便面下半包(F)", "Guorong Ma", "Jasper", "杜亚朔", "黄腾", "蒋未雨", 'wk', "蜻蜓三点水", "星际宇航员"]
    for i in range(50):
        try:
            import random 
            random.shuffle(players)
            rounds = generate_schedule(players, court_count=5, start_hour=17, elo_threshold=50, game_per_player=4, team_elo_diff=200, max_opponent_frequency=2, min_expected_wins=1)
        except ValueError as e:
            print(f"Error generating schedule: {e}")

    # all_players = [
    #     "cbt", "Yunjie", "曹大", "张晴川", "ai(F)", "Jing(F)", "Damien", "Plastic", "Jieling(F)", 
    #     "dianhsu", "Max", "gdc", "MFive(F)", "🐟🍃", "卷卷(F)", "yy(F)", "乌拉乌拉", 
    #     "米兰的小铁匠", "敏敏子(F)", "尼古丁", "一顿饭", "疏朗(F)", "z", "杨昆",
    #  "星际宇航员", "李娜(F)", "颜若儒(F)", "simonBW", "安元植", "熊猫",  "liyu", "Chao",
    #  "destiny(F)", "李东勇",  'JianjunLv', "Yummy(F)", "王威", "Louis", "毛艺钧", 
    # "方文", "shuya(F)", "Acaprice", "廖俊杰", "ian", "大米",  "Jensen", "OwenWei", "随便起个名(F)", "郑旭明", "墨欸莓(F)", "四石"
    # ]
