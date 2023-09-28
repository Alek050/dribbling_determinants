import numpy as np
import pandas as pd
from typing import Union

from calculate_angles import get_angles

def get_pressure_on_player(
    df: pd.DataFrame,
    player: str,
    *,
    pitch_size: tuple,
    d_front: Union[float, str] = "variable",
    d_back: float = 3,
    q: float = 1.75,
) -> np.ndarray:
    """
    Calculates the pressure on a single player in accordance with "Visual Analysis of Pressure
    in Soccer", Adrienko et al (2016).

    Args:
        df: A pandas DataFrame with data of all players of a frame.
        player: A string representing the column name of the player to analyze.
        pitch_size: A tuple representing the length and width of the pitch.
        d_front: A numeric or string representing the distance in meters of the front of the pressure oval.
                  If "variable", d_front will be variable based on the location on the field from the article of Mat Herold et al (2022).
                  Default is "variable".
        d_back: A numeric representing the distance in meters of the back of the pressure oval. Default is 3.
        q: A numeric representing the quotient of how fast pressure should increase/decrease as distance to the player changes. Default is 1.75.

    Returns:
        A numpy array with pressure on player over the length of the df.
    """
    if d_front == "variable":
        d_front = calculate_variable_dfront(df, player, pitch_length=pitch_size[0])
    else:
        d_front = np.array([d_front] * len(df))

    if isinstance(df, pd.Series):
        df = df.to_frame().T
        for col in [x for x in df.columns if "_x" in x or "_y" in x]:
            df[col] = df[col].astype("float64")

    team = player[:4]
    opponent_team = "away" if team == "home" else "home"
    tot_pressure = np.zeros(len(df))

    for opponent in [x[:-2] for x in df.columns if opponent_team in x and "_x" in x]:

        x_diff = df[opponent + "_x"] - df[player + "_x"]
        y_diff = df[opponent + "_y"] - df[player + "_y"]
        distance_to_player = np.nan_to_num(
            np.sqrt(x_diff**2 + y_diff**2), nan=np.inf
        )
        if (distance_to_player > max(np.append(d_front, d_back))).all():
            continue
        z = calculate_z(df, player, opponent, pitch_length=pitch_size[0])
        L = calculate_L(d_back, d_front, z)
        pressure = np.nan_to_num(
            (1 - distance_to_player / L).clip(0) ** q * 100, nan=0.0
        )
        tot_pressure += pressure
        
    return tot_pressure


def calculate_variable_dfront(
    df: pd.DataFrame,
    player: str,
    max_d_front: int = 9,
    pitch_length: float = 105.0,
) -> np.ndarray:
    """
    Calculates d_front over time according to the article of Herold et al 2022
    "Off-ball behavior in association football: A data-driven model to measure changes in
    individual defensive pressure".

    Args:
        df: A pandas DataFrame with tracking data in which the home team plays to the right.
        player: A string representing the column name of the player of which to calculate the pressure.
        max_d_front: An integer representing the maximal d_front. Default is 9.
        pitch_length: A numeric representing the length (x-direction) of the pitch. Default is 105.0.

    Returns:
        A numpy array with d_front values.
    """
    team = player[:4]
    goal_loc = (pitch_length / 2, 0) if team == "home" else (-pitch_length / 2, 0)
    x_diff = goal_loc[0] - df[player + "_x"]
    y_diff = goal_loc[1] - df[player + "_y"]
    d_to_goal = np.array(np.sqrt(x_diff**2 + y_diff**2))

    return max_d_front - 0.05 * (pitch_length - d_to_goal)


def calculate_z(
    df: pd.DataFrame,
    player: str,
    opponent: str,
    pitch_length: float = 105.0,
) -> np.ndarray:
    """
    Calculates the z in accordance with the article of Adrienko et al (2016).
    Note that the formula in the article contains a mistake, therefore the formula is not
    z = (1 - cos(phi))/2, but z = (1 + cos(phi))/2. When phi is defined as the angle between the
    vector of the player to the target (goal), and the vector of the opponent to the
    player.

    Args:
        df: A pandas DataFrame with tracking data in which the home team plays to the right.
        player: A string representing the column name of the player of which to calculate the pressure.
        opponent: A string representing the column name of the player which is pressuring the player.
        pitch_length: A numeric representing the length (x-direction) of the pitch. Default is 105.0.

    Returns:
        A numpy array with z values.
    """
    team = player[:4]

    # create tuples with x and y position of the goal, opponent and player
    goal_loc = (pitch_length / 2, 0) if team == "home" else (-pitch_length / 2, 0)
    opponent_loc = (df[opponent + "_x"].to_numpy(), df[opponent + "_y"].to_numpy())
    player_loc = (df[player + "_x"].to_numpy(), df[player + "_y"].to_numpy())

    # create vector between player and the goal
    player_goal_x_diff = goal_loc[0] - player_loc[0]
    player_goal_y_diff = goal_loc[1] - player_loc[1]
    target_vecs = np.array(
        [[x, y] for x, y in zip(player_goal_x_diff, player_goal_y_diff)]
    )

    # create vector between the player and the opponent
    player_opponent_x_diff = opponent_loc[0] - player_loc[0]
    player_opponent_y_diff = opponent_loc[1] - player_loc[1]
    pressure_vecs = np.array(
        [[x, y] for x, y in zip(player_opponent_x_diff, player_opponent_y_diff)]
    )

    angles = get_angles(target_vecs, pressure_vecs, format="radian")
    return (1.0 + np.cos(angles)) / 2.0


def calculate_L(d_back: float, d_front: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Calculates the L of the formula in accordance with Adrienko et al. (2016).

    Args:
        d_back: A numeric representing the maximal distance to back from where pressure can be measured.
        d_front: A numpy array representing the maximal distance in front of which pressure can be measured.
        z: A numpy array representing the z values in accordance with formulas in Adrienko et al (2016).

    Returns:
        A numpy array with L values.
    """

    return (d_back + (d_front - d_back) * ((z**3 + 0.3 * z) / 1.3)).clip(0.0001)
