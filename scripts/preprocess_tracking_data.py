import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
from calculate_angles import get_angle

def preprocess_tracking_data(tracking_data:pd.DataFrame, Fs: int=10, save_name: str="preprocessed_positions") -> pd.DataFrame:
    """Preprocess the tracking data to compute the velocity and acceleration of each player, 
    the individual ball possessions and duels.

    Args:
        tracking_data (pd.DataFrame): Tracking data of a match, formatted by the function reformat_positions.
        Fs (int, optional): sample frequency of the tracking_data. Defaults to 10.
        save_name (str, optional): name of the preprocessed tracking_data file. Defaults to "preprocessed_positions".

    Returns:
        pd.DataFrame: preprocessed tracking data
    """

    # make a copy of the tracking data to not overwrite the original
    tracking_data_to_process = tracking_data.copy()
    
    # represent the home team as playing from left to right and the
    # away team as playing from right to left for the full match
    tracking_data_to_process = normalize_playing_direction_tracking_data(tracking_data_to_process)

    # filter the ball data
    tracking_data_to_process[["ball_x", "ball_y"]] = filter_data(
        tracking_data_to_process[["ball_x", "ball_y"]],
        filter_type="savitzky_golay",
        window_length=7,
        polyorder=2,
    )

    # compute the velocity and acceleration of each player
    tracking_data_to_process = differentiate(
        tracking_data_to_process,
        metric="",  # no prefix so differentiate just the x and y values
        new_name="velocity",
        filter_type="savitzky_golay",
        window=7,
        poly_order=2,
        Fs=Fs,
        max_val=12,  # maximum speed of 12 m/s for each player
    )
    tracking_data_to_process = differentiate(
        tracking_data_to_process,
        metric="v",  # differentiate the velocities
        new_name="acceleration",
        filter_type="savitzky_golay",
        window=25,
        poly_order=2,
        Fs=Fs,
        max_val=7,  # maximum accleration of 7 m/s/s for each player
    )

    # compute the individual ball possessions and duels
    possession_duels = get_player_possessions_and_duels(
        tracking_data_to_process, pz_radius=1.5, dz_radius=1.5
    )
    tracking_data_to_process["player_possession"] = possession_duels[0]
    tracking_data_to_process["duels"] = possession_duels[1]

    # save preprocessed data
    tracking_data_to_process.to_pickle(f"data/{save_name}.pickle")

    return tracking_data_to_process

def normalize_playing_direction_tracking_data(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the playing direction of the home and away team for the full match.
    The home team should be playing from left to right and the away team from right to left 
    for the full match. In this dataset of unprocessed positions, both teams are represented
    by playing from left to right and the ball from right to left (don't ask me why).

    Args:
        tracking_data (pd.DataFrame): Tracking data of a match, formatted by the function reformat_positions.

    Returns:
        pd.DataFrame: Tracking data with normalized playing direction.
    """

    away_team_cols = [col for col in tracking_data.columns if col.startswith("away")]
    tracking_data[away_team_cols] *= -1
    tracking_data[["ball_x", "ball_y"]] *= -1
    return tracking_data

def differentiate(df: pd.DataFrame, *, metric: str="", new_name: str, Fs: int = 25, filter_type: str = "savitzky_golay", window: int = 7, max_val: float = float("inf"), poly_order: int = 2) -> pd.DataFrame:
    """
    Calculates the numerical differentiation of the specified metric in x and y direction and 
    updates the DataFrame with the differentiated values in x and y direction and the 
    absolute magnitude.

    Args:
        df: A pandas DataFrame with position data in x and y direction of players and ball.
        metric: A string representing the metric to differentiate the value on.
            Note that f"{player}_{metric}x" and y should exist.
        new_name: A string representing the name of the magnitude. The first letter will 
            be used for the x and y directions: f"{player}_vx" and f"{player}_velocity" if 
            new_name = "velocity".
        Fs: An integer representing the sample frequency of the data. Default is 25.
        filter_type: A string representing which filter to use: {"moving_average", 
            "savitzky_golay", None}. Default is "savitzky_golay".
        window: An integer representing the window of the filter. Default is 7.
        max_val: A float representing the maximum value of the differentiated value. 
            For instance, a speed of higher than 12 m/s is very unlikely. Default is infinity.
        poly_order: An integer representing the polynomial for the Savitzky-Golay filter. 
            Default is 2.

    Returns:
        A pandas DataFrame with the differentiated columns added.
    """

    assert filter_type in ["moving_average", "savitzky_golay", None]

    to_skip = len(metric) + 2
    col_names = [x[:-to_skip] for x in df.columns if f"_{metric}x" in x]

    dt = 1.0 / Fs

    for player in col_names:
        diff_x = df[player + f"_{metric}x"].diff() / dt
        diff_y = df[player + f"_{metric}y"].diff() / dt

        # remove outliers
        raw_differentiated = np.sqrt(diff_x**2 + diff_y**2)
        if player != "ball":
            diff_x[(raw_differentiated > max_val) & (diff_x > max_val)] = max_val
            diff_y[(raw_differentiated > max_val) & (diff_y > max_val)] = max_val

        # smoothing the signal
        if filter is not None:
            diff_x = filter_data(
                diff_x,
                filter_type=filter_type,
                window_length=window,
                polyorder=poly_order,
            )
            diff_y = filter_data(
                diff_y,
                filter_type=filter_type,
                window_length=window,
                polyorder=poly_order,
            )

        df[player + f"_{new_name[0]}x"] = diff_x
        df[player + f"_{new_name[0]}y"] = diff_y
        df[player + f"_{new_name}"] = np.sqrt(diff_x**2 + diff_y**2)
        df = df.copy()
    return df

def filter_data(df: pd.DataFrame, filter_type: str = "savitzky_golay", window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    """
    Filters data in a pandas DataFrame. Can either use a Savitzky-Golay filter or a simple 
    moving average.

    Args:
        df: A pandas DataFrame with all data that needs to be filtered.
        filter_type: A string representing the kind of filter that should be applied. 
            {"savitzky_golay", "moving_average"}. Default is "savitzky_golay".
        window_length: An integer representing the window length of the filter. Default is 7.
        polyorder: An integer representing the order of the polynomial, only for the 
            Savitzky-Golay filter. Default is 2.

    Returns:
        A numpy array in the same shape of the DataFrame, with the filtered data.
    """

    assert len(df) > window_length, "length of data should be > than window length!"

    if filter_type == "savitzky_golay":
        return savgol_filter(
            df, axis=0, window_length=window_length, polyorder=polyorder
        )
    elif filter_type == "moving_average":
        return df.rolling(window_length).mean()

def get_player_possessions_and_duels(
        df_tracking: pd.DataFrame, 
        *, 
        pz_radius: float = 1.0, 
        dz_radius: float = 1.0, 
        bs_threshold: float = 5.0, 
        ba_threshold: int = 10, 
        bd_threshold: float = 0.1, 
        min_frames: int = 0
    ) -> tuple:
    """
    Calculates which player has possession of the ball in accordance with the 
    article of Vidal-Codina et al (2022): "Automatic Event Detection in Football 
    Using Tracking Data". The algorithm finds if the ball is within the possession 
    zone (PZ) of a player, it loses the ball when the ball is outside of the PZ in 
    a next frame and a different player gains the ball in his PZ. The ball is only 
    gained when the ball changes direction or speed, to correct for balls that fly 
    over players.

    Args:
        df_tracking: A pandas DataFrame with tracking data over which to calculate the 
            player possession.
        pz_radius: A float representing the radius of the possession zone of all players. 
            Default is 1.0.
        dz_radius: A float representing the radius around ball of the duel zone. 
            Default is 1.0.
        bs_threshold: A float representing the ball speed velocity (m/s) change needed 
            to gain ball possession. Default is 5.0.
        ba_threshold: An integer representing the ball angle (degrees) change needed to gain 
            ball possession. Default is 10.
        bd_threshold: A float representing the ball displacement change needed to lose ball 
            possession. Default is 0.1.
        min_frames: An integer representing the minimal number of frames a player should be close 
            to the ball to get a potential ball possession assigned. Default is 0.

    Returns:
        A tuple containing two ndarrays: one with which player has possession and one with duels over time.
    """

    df_bp = pd.DataFrame(index=df_tracking.index)  # dataframe with ball possession info
    pz_initial, duel_initial = get_pz_and_duels(
        df_tracking, pz_radius=pz_radius, dz_radius=dz_radius, min_frames=min_frames
    )
    df_bp["pz_initial"] = pz_initial
    df_bp["duel_players"] = duel_initial
    df_bp["player_possession"] = None
    df_bp.loc[:, "ball_velocity_change"] = df_tracking.loc[:, "ball_velocity"].diff()

    start_idx = df_bp.index[0]
    end_idx = df_bp.index[0]

    while end_idx < df_bp.index[-1]:
        if pd.isnull(df_bp.loc[start_idx, "pz_initial"]):
            next_play_idx = np.where(
                ~np.isin(df_bp.loc[start_idx:, "pz_initial"], [None])
            )[0]
            if len(next_play_idx) > 0:
                start_idx = df_bp.iloc[next_play_idx[0]].name
                end_idx = start_idx
            else:
                end_idx = df_bp.index[-1]
        else:
            player = df_bp.loc[start_idx, "pz_initial"]

            # find the index when the next value in 'pz_initial' is not player or None
            end_points = np.where(
                ~np.isin(df_bp.loc[start_idx:, "pz_initial"], [player, None])
            )[0]
            if len(end_points) > 0:
                end_idx = end_points[0] + start_idx - 1
            else:
                end_idx = df_bp.index[-1]

            has_possession = check_if_possession_is_gained(
                df_tracking.loc[start_idx:end_idx],
                df_bp.loc[start_idx:end_idx],
                bs_threshold=bs_threshold,
                ba_threshold=ba_threshold,
            )
            if has_possession:
                possession_end_idx = check_when_possession_is_lost(
                    df_tracking.loc[start_idx:end_idx],
                    df_bp.loc[start_idx:end_idx],
                    bd_threshold=bd_threshold,
                )

                # check if this is the same player as a previous possession

                if (
                    df_bp.loc[:start_idx, "player_possession"].last_valid_index()
                    is not None
                ):
                    last_possession_index = df_bp.loc[
                        :start_idx, "player_possession"
                    ].last_valid_index()
                    last_possession_player = df_bp.loc[
                        last_possession_index, "player_possession"
                    ]
                    if last_possession_player == player:
                        start_idx = last_possession_index

                df_bp.loc[start_idx:possession_end_idx, "player_possession"] = player

            start_idx = end_idx + 1
            end_idx = start_idx

    alive = df_bp[df_tracking["ball_status"] == "alive"]
    player_possessions = np.empty(len(df_bp), dtype="object")
    player_possessions[alive.index - df_bp.index[0]] = alive["player_possession"].values
    duels = np.empty(len(df_bp), dtype="object")
    duels[alive.index - df_bp.index[0]] = alive["duel_players"].values

    return player_possessions, duels



def get_pz_and_duels(df: pd.DataFrame, *, pz_radius: float, dz_radius: float, min_frames: int = 3) -> tuple:
    """
    Calculates the control frames. A control frame is either a possession or a duel. 
    A possession is when the ball is within the possession zone of a player. A duel is 
    when two opponents are within the duel zone radius of the ball.

    Args:
        df: A pandas DataFrame with tracking data of players and the ball.
        pz_radius: A float representing the radius around a player for a possible possession 
            in meters.
        dz_radius: A float representing the radius around the ball for a possible duel in meters.
        min_frames: An integer representing the minimum number of frames a player should be 
            close to the ball to get a potential possession assigned. Default is 3.

    Returns:
        A tuple containing two numpy arrays: one with possessions and one with duels.
    """

    temp_df = pd.DataFrame(index=df.index)
    possession_zone = np.empty(len(df), dtype="object")
    duel_zone = np.empty(len(df), dtype="object")
    start_idx = df.index[0]

    # calculate the displacement between each player and the ball
    players = [x[:-2] for x in df.columns if "_x" in x and "ball" not in x]
    for player in players:
        x_diff = df[player + "_x"] - df["ball_x"]
        y_diff = df[player + "_y"] - df["ball_y"]
        temp_df[player] = np.sqrt(x_diff**2 + y_diff**2)

    # Find possessions
    closest = temp_df.idxmin(axis=1)
    possessions = closest[temp_df.min(axis=1) <= pz_radius]
    possession_zone[possessions.index - start_idx] = possessions.values

    start_idx = 0
    end_idx = 0
    while end_idx < len(possession_zone):
        if possession_zone[start_idx] is None:
            start_idx += 1
            end_idx += 1
        else:
            player = possession_zone[start_idx]
            locs = np.where(possession_zone[start_idx:] != player)[0]
            if len(locs) > 0:
                end_idx = locs[0] + start_idx - 1
            else:
                end_idx = len(possession_zone) - 1
            if end_idx - start_idx + 1 < min_frames:
                possession_zone[start_idx : end_idx + 1] = None

            end_idx += 1
            start_idx = end_idx

    # Find duels
    home = temp_df[[x for x in temp_df.columns if "home" in x]]
    away = temp_df[[x for x in temp_df.columns if "away" in x]]
    home_closest = home.idxmin(axis=1)
    away_closest = away.idxmin(axis=1)
    home_duel_range = home_closest[home.min(axis=1) <= dz_radius]
    away_duel_range = away_closest[away.min(axis=1) <= dz_radius]
    home_duel_range.name = "home_duel_range"
    away_duel_range.name = "away_duel_range"
    duels = pd.concat([home_duel_range, away_duel_range], axis=1, join="inner")
    duel_zone[duels.index - df.index[0]] = (
        duels["home_duel_range"] + "-" + duels["away_duel_range"]
    )

    return possession_zone, duel_zone

def check_if_possession_is_gained(df_tracking: pd.DataFrame, df_bp: pd.DataFrame, *, bs_threshold: float, ba_threshold: float) -> bool:
    """
    Checks if, within a given period, a player gains possession of the ball. 
    Ball possession is obtained when the ball speed changes at least bs_threshold 
    m/s anywhere in the time period or when the ball changes direction 
    (> ba_threshold) between the first and the last frame of the DataFrame.

    NOTE: The period of the DataFrame is shortened so that only one player can potentially 
    gain the ball. Therefore, df_bp["pz_initial"].unique() has a len() of <= 2. Both the 
    player and None can be in the DataFrame.

    Args:
        df_tracking: A pandas DataFrame with tracking data.
        df_bp: A pandas DataFrame with additional, temporary information about the game.
        bs_threshold: A float representing the minimal change in ball speed needed to 
            assign a possession.
        ba_threshold: A float representing the minimal ball angle change in degree 
            needed to assign a possession.

    Returns:
        A boolean value: True if a new possession is started, False otherwise.
    """
    assert len(df_bp["pz_initial"].unique()) <= 2, "Only one player allowed in df_bp!"
    assert "ball_vx" in df_tracking.columns and "ball_vy" in df_tracking.columns

    in_possession_df = df_bp[~pd.isnull(df_bp["pz_initial"])]
    if (in_possession_df["ball_velocity_change"].abs() > bs_threshold).any():
        return True
    elif (
        df_tracking.loc[in_possession_df.index, ["ball_vx"]].first_valid_index()
        is not None
    ):
        start_idx = df_tracking.loc[
            in_possession_df.index, ["ball_vx"]
        ].first_valid_index()
        end_idx = df_tracking.loc[in_possession_df.index, ["ball_vx"]].last_valid_index()

        first_direction = df_tracking.loc[start_idx, ["ball_vx", "ball_vy"]].to_numpy()
        last_direction = df_tracking.loc[end_idx, ["ball_vx", "ball_vy"]].to_numpy()

        if get_angle(first_direction, last_direction, format="degree") > ba_threshold:
            return True
    return False


def check_when_possession_is_lost(df_tracking: pd.DataFrame, df_bp: pd.DataFrame, *, bd_threshold: float) -> int:
    """
    Checks when a player has lost possession of the ball, in accordance with the 
    article of Vidal-Codina et al (2022). The possession of the ball is lost when two 
    conditions are satisfied: (1) the ball is outside the pz_radius of the player and 
    the ball moves above the displacement threshold, and (2), the player is not present 
    in the next frame where there is either a possession or a duel.

    NOTE: The period of the DataFrame is shortened so that only one player can potentially 
    lose the ball. Therefore, df_bp["pz_initial"].unique() has a len() of <= 2. Both the 
    player and None can be in the DataFrame. Also, the next frame (outside of the shortened 
    DataFrame) is never a potential possession or duel with the same player.

    Args:
        df_tracking: A pandas DataFrame with tracking data.
        df_bp: A pandas DataFrame with additional, temporary information about the game.
        bd_threshold: A float representing the minimal displacement of the ball needed 
            to lose possession.

    Returns:
        An integer representing the index where possession is lost. If no index is found, 
        the last index of df_bp is returned.
    """

    assert len(df_bp["pz_initial"].unique()) <= 2, "Only one player allowed in df_bp!"
    assert (df_tracking.index == df_bp.index).all(), "Indexes should be alike"

    makes_bs_threshold = df_tracking["ball_velocity"] > bd_threshold
    first_invalid_idx = df_bp["pz_initial"].last_valid_index() + 1

    if makes_bs_threshold.loc[first_invalid_idx:].sum() > 0:
        return makes_bs_threshold.loc[first_invalid_idx:].idxmax() - 1
    else:
        return df_bp.index[-1]

if __name__ == "__main__":
    if "reformated_positions.csv" not in os.listdir("data"):
        from reformat_positions import reformat_positions
        reformat_positions()
    
    processed = preprocess_tracking_data(pd.read_csv("data/reformated_positions.csv", index_col=0))
    import pdb; pdb.set_trace()



