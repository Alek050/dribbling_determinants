import pandas as pd
import numpy as np
from scripts.calculate_pressure import get_pressure_on_player
from scripts.calculate_voronoi_area import get_voronoi_area
from scripts.calculate_angles import get_angles

from typing import Dict
import pandas as pd
import numpy as np

def get_features(tracking_data: pd.DataFrame, start_possession_idx: int, pitch_length: float = 105.0, pitch_width: float = 68.0) -> Dict[str, float]:
    """
    Calculates a dictionary of features for a given possession in a soccer match.

    Args:
        tracking_data: A pandas DataFrame with tracking data over the entire match.
        start_possession_idx: An integer representing the index of the start of the possession.
        pitch_length: A float representing the length of the soccer pitch. Default is 105.0.
        pitch_width: A float representing the width of the soccer pitch. Default is 68.0.

    Returns:
        A dictionary of features for the given possession.

    The function takes a pandas DataFrame `tracking_data` with tracking data over the entire match, 
    an integer `start_possession_idx` representing the index of the start of the possession, and two 
    optional floats `pitch_length` and `pitch_width` representing the length and width of the soccer 
    pitch, respectively. The function returns a dictionary of features for the given possession.

    The function first calculates the start of the duel, i.e. the first time the two players are 
    close to each other. It uses the `duels` column in the tracking data to do this. It then 
    calculates three different DataFrames for the possession phase, the duel phase, and the full phase.

    The function then calculates a number of individual, environmental, and interaction factors. 
    Individual factors include the maximum acceleration and velocity for both the attacking and 
    defending players. Environmental factors include the distance of the ball to the goal line and 
    sideline at the start of the possession and the start of the duel. Interaction factors include 
    the minimum, median, and standard deviation of the relative distance between the attacking and 
    defending players and between the defending player and the ball, as well as the maximum pressure 
    on the attacking player during the duel phase. The function also calculates the maximum relative 
    speed difference and the maximum relative moving angle between the attacking and defending 
    players during the possession phase, the duel phase, and the full phase. Finally, the function 
    calculates the action space at the start of the duel, the difference between the standard 
    deviation of the distance of the ball to the attacking player and the defending player, and the 
    median directional change of the attacking player during the full phase.

    The function returns a dictionary with string keys and float values representing the calculated 
    features.
    """
    # first calculate the start of the duel, i.e. the first time the two players are close to each other
    # we will use the `duels` column in the tracking data to do this.

    attacking_player = tracking_data.loc[start_possession_idx]["player_possession"]
    start_duel_idx = end_possession_idx = tracking_data.loc[start_possession_idx:]["duels"].first_valid_index()
    duel_players = tracking_data.loc[start_duel_idx]["duels"]
    new_duel_mask = (tracking_data["duels"] != duel_players) & (tracking_data.index > start_duel_idx)
    end_duel_idx = tracking_data.loc[new_duel_mask]["duels"].index[0]
    
    players = duel_players.split("-")
    defending_player = players[0] if players[0] != attacking_player else players[1]

    # calculate df for different phases
    possession_phase_df = tracking_data.loc[start_possession_idx:end_possession_idx]
    duel_phase_df = tracking_data.loc[start_duel_idx:end_duel_idx]
    full_phase_df = tracking_data.loc[start_possession_idx:end_duel_idx]

    res_dict = {}
    # individual factors
    res_dict["max_acc_att_full_action"] = full_phase_df[f"{attacking_player}_acceleration"].max()
    res_dict["max_acc_def_full_action"] = full_phase_df[f"{defending_player}_acceleration"].max()
    res_dict["med_acc_att_full_action"] = full_phase_df[f"{attacking_player}_acceleration"].median()
    res_dict["max_vel_att_full_action"] = full_phase_df[f"{attacking_player}_velocity"].max()
    res_dict["max_vel_def_full_action"] = full_phase_df[f"{defending_player}_velocity"].max()

    # environmental factors
    goal_loc = np.array([pitch_length/2., 0]) if attacking_player.startswith("home") else np.array([-pitch_length/2., 0])
    ball_loc_start_possession = np.array([possession_phase_df.iloc[0]["ball_x"], possession_phase_df.iloc[0]["ball_y"]])
    ball_loc_start_duel = np.array([duel_phase_df.iloc[0]["ball_x"], duel_phase_df.iloc[0]["ball_y"]])
    res_dict["dist_ball_goalline_start_possession"] = np.abs(goal_loc[0] - ball_loc_start_possession[0])
    res_dict["dist_ball_goalline_start_duel"] = np.abs(goal_loc[0] - ball_loc_start_duel[0])
    res_dict["dist_ball_sideline_start_possession"] = np.abs(pitch_width/2. - np.abs(ball_loc_start_possession[1]))
    res_dict["dist_ball_sideline_start_duel"] = np.abs(pitch_width/2. - np.abs(ball_loc_start_duel[1]))

    # interaction factors
    attacker_position = full_phase_df.loc[:, [f"{attacking_player}_x", f"{attacking_player}_y"]]
    defender_position = full_phase_df.loc[:, [f"{defending_player}_x", f"{defending_player}_y"]]
    ball_position = full_phase_df.loc[:, ["ball_x", "ball_y"]]
    dist_att_def = np.linalg.norm(attacker_position.values - defender_position.values, axis=1)
    dist_att_ball = np.linalg.norm(attacker_position.values - ball_position.values, axis=1)
    dist_def_ball = np.linalg.norm(defender_position.values - ball_position.values, axis=1)
    pressure_duel = get_pressure_on_player(duel_phase_df, attacking_player, pitch_size=[pitch_length, pitch_width])
    attacker_space = get_voronoi_area(full_phase_df.loc[start_duel_idx], attacking_player, field_dim=[pitch_length, pitch_width])
    defender_space = get_voronoi_area(full_phase_df.loc[start_duel_idx], defending_player, field_dim=[pitch_length, pitch_width])
    directional_change_att = np.arctan2(duel_phase_df[f"{attacking_player}_vy"]*duel_phase_df[f"{attacking_player}_ax"] - duel_phase_df[f"{attacking_player}_vx"]*duel_phase_df[f"{attacking_player}_ay"],
                                    duel_phase_df[f"{attacking_player}_vy"]*duel_phase_df[f"{attacking_player}_ay"] + duel_phase_df[f"{attacking_player}_vx"]*duel_phase_df[f"{attacking_player}_ax"])
    directional_change_att = np.rad2deg(directional_change_att)

    res_dict["min_relative_distance_att_def"] = dist_att_def.min()
    res_dict["std_relative_distance_att_def"] = dist_att_def.std()
    res_dict["median_relative_distance_att_def"] = np.median(dist_att_def)
    res_dict["std_relative_distance_att_ball"] = dist_att_ball.std()
    res_dict["std_relative_distance_def_ball"] = dist_def_ball.std()
    res_dict["min_relative_distance_def_ball"] = dist_def_ball.min()
    res_dict["max_pressure_att_duel_phase"] = np.max(pressure_duel)
    res_dict["max_relative_speed_difference_att_def_possession_phase"] = np.max(np.abs(possession_phase_df[f"{attacking_player}_velocity"] - possession_phase_df[f"{defending_player}_velocity"]))
    res_dict["median_relative_speed_difference_att_def_full_action"] = np.median(np.abs(full_phase_df[f"{attacking_player}_velocity"] - full_phase_df[f"{defending_player}_velocity"]))
    res_dict["max_relative_speed_difference_att_def_duel_phase"] = np.max(np.abs(duel_phase_df[f"{attacking_player}_velocity"] - duel_phase_df[f"{defending_player}_velocity"]))
    res_dict["max_relative_moving_angle_att_def_duel_phase"] = get_angle_diff_in_metric(duel_phase_df, attacking_player, second=defending_player, metric="v", func=np.max)
    res_dict["max_speed_angle_interaction_att_def_duel_phase"] = get_max_speed_angle_interaction(duel_phase_df, attacking_player, second=defending_player)
    res_dict["action_space_start_duel"] = attacker_space + defender_space
    res_dict["std_ball_att_std_ball_def_full_action"] = np.std(dist_att_ball) - np.std(dist_def_ball)
    res_dict["median_directional_change_att_full_action"] = np.median(np.abs(directional_change_att))

    return res_dict

def get_angle_diff_in_metric(df: pd.DataFrame, player: str, second: str = "ball", metric: str = "v", func: callable = np.max) -> float:
    """
    Calculates the maximum difference in moving angle between the attacking player and the second player or the ball
    in the specified metric.

    Args:
        df: A pandas DataFrame with tracking data over a short time frame.
        player: A string representing the column of the attacking player.
        second: A string representing the column of the second player or the ball. Default is "ball".
        metric: A string representing the metric to use. Either "v" (velocity) or "a" (acceleration). Default is "v".
        func: A callable function to apply to the angle differences. Default is np.max.

    Returns:
        A float representing the maximal difference in moving angle between the attacking player and the second player or the ball.
    """

    assert metric in ["v", "a"]

    player_loc = df[[f"{player}_{metric}x", f"{player}_{metric}y"]].to_numpy()
    second_loc = df[[f"{second}_{metric}x", f"{second}_{metric}y"]].to_numpy()
    angles = get_angles(player_loc, second_loc, format="degree")
    return func(np.abs(angles))

def get_max_speed_angle_interaction(df: pd.DataFrame, player: str, second: str = "ball") -> float:
    """
    Calculates the maximum of (speed * angle), where the speed is the difference in absolute speed and the angle is the
    absolute difference in moving angle between the attacking player and the second player or the ball.

    Args:
        df: A pandas DataFrame with tracking data over a short time frame.
        player: A string representing the column of the attacking player.
        second: A string representing the column of the second player or the ball. Default is "ball".

    Returns:
        A float representing the maximal difference in velocity times the moving angle.
    """
    player_loc = df[[f"{player}_vx", f"{player}_vy"]].to_numpy()
    second_loc = df[[f"{second}_vx", f"{second}_vy"]].to_numpy()
    angles = get_angles(player_loc, second_loc, format="radian")
    vel_diff = np.abs(df[f"{player}_velocity"] - df[f"{second}_velocity"])
    interaction = vel_diff * angles

    return np.max(interaction)






