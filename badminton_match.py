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
    check_play_times(players, rounds_lineups, game_per_player=game_per_player)
    check_partnerships(rounds_lineups)
    check_gender_balance(rounds_lineups)
    check_elo_balance(rounds_lineups, player_elos, elo_threshold)
    check_opponent_frequency(rounds_lineups, max_opponent_frequency=game_per_player // 2)
    check_consecutive_rounds(players, rounds_lineups, 4)
    output_file = f"badminton_schedule_{elo_threshold}_{team_elo_diff}.xlsx"
    # Save the schedule to Excel
    save_schedule_to_excel(rest_schedule, rounds_lineups, output_file, start_hour=start_hour)
    return rounds_lineups 


if __name__ == "__main__":
    players = [ "敏敏子(F)", "Acaprice", 'liyu', "Max", "张晴川",  "方文", "米兰的小铁匠",  "gdc", 
               "Jensen", "一顿饭", "曹大", "Louis", "杨昆", "Jieling(F)", "Damien", "Plastic", 
               "cbt", "Yummy(F)", "ai(F)", "青天烟云", "郑旭明", "Jing(F)", "墨欸莓(F)", "四石"]
    # sort players by ELO rating
    # players.sort(key=lambda x: load_existing_player_data()[0].get(x, 0), reverse=True)
    # print(f"Sorted players by ELO: {players}")
    for i in range(100):
        try:
            import random 
            random.shuffle(players)
            rounds = generate_schedule(players, court_count=4, start_hour=14, elo_threshold=70, game_per_player=6, team_elo_diff=300)
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
