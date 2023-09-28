import pandas as pd
import numpy as np
from tqdm import tqdm

def reformat_positions(first_half_csv: str ="data/positions_FIRST_HALF.csv", second_half_csv: str="data/positions_SECOND_HALF.csv") -> None:
    """Reformat the information from a long format to a wider format. The resulting dataframe will 
    have the following columns:
    - ball_x: x coordinate of the ball
    - ball_y: y coordinate of the ball
    - ball_status: status of the game (alive or dead)
    - ball_possession: team that is in possession of the ball (home or away)
    - period: period id of the game (1 (first half) or 2 (second half))
    - seconds: time in seconds of the game
    - f"{team}_{player_id}_{metric}": team is either 'home' or 'away', player_id is the id of the player
        metric is either 'x', 'y', 'velocity' or 'acceleration' representing the x coordinate, y coordinate, 
        velocity and acceleration of the player.

    Args:
        first_half_csv (str, optional): Location to the first half positions csv file. 
            Defaults to "data/positions_FIRST_HALF.csv".
        second_half_csv (str, optional): Location to the second half positions csv file. 
            Defaults to "data/positions_SECOND_HALF.csv".
    """
    first_half_df = pd.read_csv(first_half_csv, index_col=0)
    second_half_df = pd.read_csv(second_half_csv, index_col=0)

    first_half_players = first_half_df.groupby("group_id")["player_id"].unique()
    first_half_home_players = first_half_players.loc[1] # 1 is home team id
    first_half_away_players = first_half_players.loc[2] # 2 is away team id

    second_half_players = second_half_df.groupby("group_id")["player_id"].unique()
    second_half_home_players = second_half_players.loc[1] # 1 is home team id
    second_half_away_players = second_half_players.loc[2] # 2 is away team id

    # compute  unique home and away players for the full match
    home_players = np.concatenate([first_half_home_players, [x for x in second_half_home_players if x not in first_half_home_players]])
    away_players = np.concatenate([first_half_away_players, [x for x in second_half_away_players if x not in first_half_away_players]])
    all_players = np.concatenate([home_players, away_players])

    # map possessions and alive flags to strings
    possession_map = {1: "home", 2: "away"}
    alive_map = {0: "dead", 1: "alive"}

    # create a dictionary with all the columns we want to have in the final dataframe
    res_dict = {"ball_x": [], "ball_y": [], "ball_status": [], "ball_possession": [], "period": [], "seconds": []}
    for team, players in [("home", home_players), ("away", away_players)]:
        for player in players:
            for metric in ["x", "y", "velocity", "acceleration"]:
                res_dict[f"{team}_{player}_{metric}"] = []

    # iterate over the two halves and create the final dataframe
    for i, half_df in enumerate([first_half_df, second_half_df]):
        grouped_df = half_df.groupby("timestamp_ms")
        for time_ms, group in tqdm(grouped_df):
            res_dict["seconds"].append(time_ms / 1000)
            res_dict["period"].append(i + 1)
            res_dict["ball_x"].append(group["x_ball"].iloc[0])
            res_dict["ball_y"].append(group["y_ball"].iloc[0])
            status_id = group["AliveFlag"].iloc[0]
            possession_id = group["PossessionFlag"].iloc[0]
            res_dict["ball_status"].append(alive_map[status_id])
            res_dict["ball_possession"].append(possession_map[possession_id])

            # add players information
            for team, players in [("home", home_players), ("away", away_players)]:
                team_group = group[group["player_id"].isin(players)]
                for player in players:
                    if player in team_group["player_id"].values:
                        player_group = team_group[team_group["player_id"] == player]
                        res_dict[f"{team}_{player}_x"].append(player_group["x"].iloc[0])
                        res_dict[f"{team}_{player}_y"].append(player_group["y"].iloc[0])
                        res_dict[f"{team}_{player}_velocity"].append(player_group["speed"].iloc[0])
                        res_dict[f"{team}_{player}_acceleration"].append(player_group["acceleration"].iloc[0])
                    else:
                        res_dict[f"{team}_{player}_x"].append(np.nan)
                        res_dict[f"{team}_{player}_y"].append(np.nan)
                        res_dict[f"{team}_{player}_velocity"].append(np.nan)
                        res_dict[f"{team}_{player}_acceleration"].append(np.nan)
    
    # create the final dataframe
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv("data/reformated_positions.csv")


if __name__ == "__main__":
    reformat_positions()
