import pandas as pd
import numpy as np
from databallpy.match import Match

class OneVsOne():
    def __init__(self, tracking_data:pd.DataFrame, attacker_column_id:str, defender_column_id:str, attacker_side:str, defender_side:str, is_successful:bool):
        self.tracking_data = tracking_data
        self.attacker_column_id = attacker_column_id
        self.defender_column_id = defender_column_id
        self.attacker_side = attacker_side
        self.defender_side = defender_side
        self.is_successful = is_successful


class TakeOnEvent():
    def __init__(self, tracking_batch:pd.DataFrame, attacker_column_id:str, is_successful:bool, match_name: str, start_second: int, normalize_direction:bool=True):
        self.tracking_data_batch = tracking_batch.copy()
        self.attacker_column_id = attacker_column_id
        self.is_successful = is_successful
        self.attacking_team = attacker_column_id[:4]
        self.match_name = match_name
        self.start_second = start_second
        if normalize_direction:
            self.normalize_direction()
        self.defenders = {}
        self.attacker_name = None
        self.attacker_team = None
        self.home_team_name = None
        self.away_team_name = None
        self.frame_rate = None
        
    def normalize_direction(self):
        """Function to normalize the playing direction of the tracking data.
        After calling this function, the attacking team is represented as moving
        From left to right.
        """
        if self.attacking_team == "away":
            cols_to_invert = [x for x in self.tracking_data_batch.columns if x[-1] == "x" or x[-1]=="y"]
            self.tracking_data_batch[cols_to_invert] *= -1
    
    def add_extra_info_to_event(self, match:Match, max_dist:float) -> None:
        """Function to add extra information to the event, such as the name of the
        attacker and the defenders that are close to him.

        Args:
            match (Match): The match of the take on event.
            max_dist (float): The number of meters a defender should be within to
            be considered close enough.
        """
        self.add_close_defenders(match, max_dist=max_dist)
        self.attacker_name = match.player_column_id_to_full_name(self.attacker_column_id)
        self.attacker_team = match.home_team_name if self.attacking_team == "home" else match.away_team_name
        self.home_team_name = match.home_team_name
        self.away_team_name = match.away_team_name
        self.frame_rate = match.frame_rate

    def add_close_defenders(self, match:Match, max_dist:float=1.5) -> dict:
        """Function that looks for all defenders that are close to the attacker during
        the take on event. Add all of them to self.defenders.

        Args:
            match (Match): The match of the take on event.
            max_dist (float, optional): The number of meters a defender should be within to
            be considered close enough. Defaults to 1.5 meters.
        """

        defending_team = "home" if self.attacking_team == "away" else "away"
        opponent_column_ids = [x[:-2] for x in self.tracking_data_batch.columns if x[:4] == defending_team and "_x" in x]
        attacker_loc = self.tracking_data_batch[[self.attacker_column_id + "_x", self.attacker_column_id + "_y"]].values 
        for opponent_column_id in opponent_column_ids:
            opponent_loc = self.tracking_data_batch[[opponent_column_id + "_x", opponent_column_id + "_y"]].values
            dist = np.linalg.norm(attacker_loc - opponent_loc, axis=1)
            if np.min(dist) < max_dist:
                self.defenders[match.player_column_id_to_full_name(opponent_column_id)] = opponent_column_id
    
    
        

def get_take_on_events(match: Match, extra_time:float, normalize_direction:bool=True) -> list:
    """Function to obtain all take on events from a match. Note that the match
    should be a preprocessed databallpy Match object. 

    Args:
        match (Match): The match to obtain the take on events from.
        extra_time (float): The number of seconds to add to the start and end of the
            take on event.
        normalize_direction (bool, optional): Whether to normalize the direction of
            the tracking data. Defaults to True.

    Returns:
        list: A list of TakeOnEvent objects for all matches in the folder.
    """

    take_ons = []
    if not match.is_synchronised:
        return []
    for dribble in match.dribble_events.values():
        if len(match.tracking_data[match.tracking_data["event_id"]==dribble.event_id]) == 0:
            # dribble not synced to tracking data
            continue
        frame_idx = match.tracking_data[match.tracking_data["event_id"]==dribble.event_id].index[0]
        start_idx = max(match.tracking_data.index[0], frame_idx - (extra_time*match.frame_rate))
        end_idx = min(match.tracking_data.index[-1], frame_idx + (extra_time*match.frame_rate) + 1)
        tracking_batch = match.tracking_data.loc[start_idx:end_idx]
        attacker_column_id = match.player_id_to_column_id(dribble.player_id)
        start_second = (frame_idx - match.tracking_data.index[0]) / match.frame_rate
        event = TakeOnEvent(
            tracking_batch, 
            attacker_column_id, 
            dribble.outcome, 
            match_name = match.name,
            start_second = start_second,
            normalize_direction=normalize_direction)
        event.add_extra_info_to_event(match, max_dist=1.5)
        take_ons.append(event)
    
    return take_ons
        