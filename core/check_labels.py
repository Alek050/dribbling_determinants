import pandas as pd
import numpy as np
import random
import os
from databallpy import get_saved_match
from core.take_on_event import get_take_on_events, OneVsOne


def check_where_labels_are_not_aligned(data_dir:str="data/labels") -> pd.DataFrame :
    """Function that looks for all files that labeled a match in data_dir. 
    For every file, there should be at least two unique raters. For every possible
    1vs1 dribble it is checked if the raters agree on validity of successfulness of the
    take on. For all matches where there is no perfectly agreement, returns a dict with 
    the match id as key and a dict with the original raters, the proposed new rater, 
    the row and if it needs to be checked for validness or succesfulness.

    Args:
        data_dir (str, optional): Location of directory where all labels are saved.
        Defaults to "data/labels". Note that the labels files should be of .xlsx format
        and the name of the file should be "{match_id} {rater_name}".

    Returns:
        pd.DataFrame: information about the take ons that need to be checked again.
    """

    # get all files in data_dir
    files = os.listdir(data_dir)
    unique_matches =  set([" ".join(file.split(" ")[:-1]) for file in files if file.endswith(".xlsx")])
    raters = set([file.split(" ")[-1].split(".")[0] for file in files if file.endswith(".xlsx")])
    res_df = pd.DataFrame()
    for match in unique_matches:
        match_raters = []
        match_labels = {}
        result = {
            "match": [],
            "original_raters": [],
            "proposed_rater": [],
            "take_on_id": [],
            "defender_num": [],
            "check_valid": [],
            "check_success": []
            }
        
        for rater_file in [file for file in files if match in file]:
            match_rater = next(name for name in raters if name in rater_file.lower())
            match_raters.append(match_rater)
            match_labels[match_rater] = pd.read_excel(f"{data_dir}/{rater_file}")

        # check if there are at least 2 raters
        if len(match_raters) < 2:
            print(f"match {match} has less than 2 raters")
            continue
        
        if len([r for r in raters if r not in match_raters]) > 0:
            new_rater = random.choice([r for r in raters if r not in match_raters])
        else:
            new_rater = "no new raters available"
        col_names = ["valid_action", "succesfulness"]
        for idx in match_labels[match_raters[0]].index:
            # delete rows where the match is not defined
            if pd.isnull(match_labels[match_raters[0]].loc[idx, "name"]):
                continue
            
            row_vals = pd.concat([pd.DataFrame(s.loc[idx, col_names]).T for s in match_labels.values()])

            # make sure that all raters have inserted either 0 or 1 for validity
            if not row_vals["valid_action"].isin([0, 1]).all():
                print(f"match {match} has a non valid value for valid_action in row {idx}")
                continue

            # check if there is consensus on if it is a valid action
            counts = row_vals["valid_action"].value_counts()
            if (len(counts) != 1 and counts.iloc[0] == counts.iloc[1]) or counts.iloc[0] == 1:
                # majority vote is not found for valid action
                result["match"].append(match)
                result["original_raters"].append(" & ".join(match_raters))
                result["proposed_rater"].append(new_rater)
                result["take_on_id"].append(match_labels[match_raters[0]].loc[idx, "take_on_id"])
                result["defender_num"].append(match_labels[match_raters[0]].loc[idx, "defending_player_num"])
                result["check_valid"].append(True)
                result["check_success"].append(True)
                # valid action and succes should be checked, thus continue
                continue

            if counts.index[0] == 0:
                # is not a valid action, succesfulness should not be checked
                continue
            
            # make sure that all raters have inserted -1, 0, 1 or np.nan
            if not row_vals["succesfulness"].isin([-1, 0, 1, np.nan]).all():
                print(f"match {match} has a non valid value for succesfulness in row {idx}")
                continue
        
            # check if there is consensus on if it is a successful action
            counts = row_vals["succesfulness"].value_counts()
            if (len(counts) != 1 and counts.iloc[0] == counts.iloc[1]) or counts.iloc[0] == 1:
                # majority vote is not found for valid action
                result["match"].append(match)
                result["original_raters"].append(" & ".join(match_raters))
                result["proposed_rater"].append(new_rater)
                result["take_on_id"].append(match_labels[match_raters[0]].loc[idx, "take_on_id"])
                result["defender_num"].append(match_labels[match_raters[0]].loc[idx, "defending_player_num"])
                result["check_valid"].append(False)
                result["check_success"].append(True)

        temp_df = pd.DataFrame(result)
        res_df = pd.concat([res_df, temp_df])
    return res_df

def get_valid_labels(data_dir:str = "data/labels") -> pd.DataFrame:
    """Function to obtain for all labeled take ons the majority vote of the raters.

    Args:
        data_dir (str, optional): location of directory where all labels are saved. Defaults to "data/labels".

    Returns:
        pd.DataFrame: for every take on the label of the majority vote of the raters.
    """
    files = os.listdir(data_dir)
    unique_matches =  set([" ".join(file.split(" ")[:-1]) for file in files if file.endswith(".xlsx")])
    df = pd.DataFrame()
    n_take_ons = 0
    n_1vs1 = 0
    

    for match_name in unique_matches:
        match_labels = {} 
        for i, rater_file in enumerate([file for file in files if match_name in file]):
            match_labels[i] = pd.read_excel(f"{data_dir}/{rater_file}")
        
        # check if there are at least 2 raters
        if len(match_labels) < 2:
            print(f"match {match_name} has less than 2 raters")
            continue
        
        take_on_id = None
        for idx in match_labels[0].index:
            # delete rows where the match is not defined
            if pd.isnull(match_labels[0].loc[idx, "name"]):
                continue
            
            # count 1vs1
            n_1vs1 += 1

            # get take on id
            if not take_on_id == match_labels[0].loc[idx, "take_on_id"]:
                n_take_ons += 1
                take_on_id = match_labels[0].loc[idx, "take_on_id"]

            col_names = ["valid_action", "succesfulness"]
            row_vals = pd.concat([pd.DataFrame(s.loc[idx, col_names]).T for s in match_labels.values()])
            
                
            # check if all inputs are valid
            if not row_vals["valid_action"].isin([0, 1]).all():
                print(f"match {match_name} has a non valid value for valid_action in row {idx}")
                continue
            
            
            # check and get majority vote for valid action
            counts = row_vals["valid_action"].value_counts()
            if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
                # majority vote is found for valid action
                validness = counts.index[0]
            else:
                print(f"No consensus on valid action in match {match_name}")
                continue
            
            if validness == 1:
                # check if there is consensus on if it is a successful action
                counts = row_vals["succesfulness"].value_counts()
                
                    # check if all inputs are valid
                if not counts.index.isin([-1, 0, 1]).all():
                    print(f"match {match_name} has a non valid value for succesfulness in row {idx}")
                    continue

                if len(counts) == 1 or counts.iloc[0] > counts.iloc[1]:
                    # majority vote is found for succesfulness
                    succesfulness = counts.index[0]
                else:
                    if len(match_labels) > 1:
                        print(f"No consensus on succesfulness in match {match_name}")
                    continue

                # add to dataframe
                row = match_labels[0].loc[idx].copy()
                row["valid_action"] = validness
                row["succesfulness"] = succesfulness
                df = pd.concat([df, pd.DataFrame(row).T])
    
    print(f"{n_take_ons} take ons and {n_1vs1} 1vs1's are labeled")

    return df

def get_valid_actions(tracking_data_dir:str, labels_dir:str = "data/labels") -> list:
    """Function to get all valid actions from the labels in data_dir
    
    Args:
        tracking_data_dir (str): location of directory where all tracking data is saved.
        data_dir (str): location of directory where all labels are saved.
        Defaults to "data/labels".
        
    returns:
        list: list with TakeOne's of all valid actions"""
    
    # get all files in data_dir
    valid_actions = []
    labels = get_valid_labels(data_dir=labels_dir)
    labels["datetime"] = pd.to_datetime(labels["datetime"]).dt.tz_localize("Europe/Amsterdam")
    files = os.listdir(labels_dir)
    unique_matches =  set([" ".join(file.split(" ")[:-1]) for file in files if file.endswith(".xlsx")])

    # save statistics
    attacking_teams = []
    defending_teams = []
    attacking_players = []
    defending_players = []
    teams_times = []

    for match_name in unique_matches: 
        match_name = match_name.split("_")[0]
        match = get_saved_match(match_name, path=tracking_data_dir)
        take_ons = get_take_on_events(match, extra_time=6.5, normalize_direction=True)
        if len(take_ons) > 0:
            teams_times.append(match.home_team_name)
            teams_times.append(match.away_team_name)
        for id, take_on in enumerate(take_ons):
            start_dt = pd.to_datetime(take_on.tracking_data_batch.iloc[0]["datetime"])
            end_dt = pd.to_datetime(take_on.tracking_data_batch.iloc[-1]["datetime"])
            mask = (
                (labels["name"] == match_name) 
                & (labels["datetime"].between(start_dt, end_dt, inclusive="both")) 
                & (labels["valid_action"] == 1)
                & (labels["attacking_player"] == take_on.attacker_name)
            )
            one_vs_ones = labels.loc[mask]
            if len(one_vs_ones) == 0:
                continue
            for _, one_vs_one in one_vs_ones.iterrows():
                defender_num = int(one_vs_one["defending_player_num"])
                defender_side = "home" if take_on.attacking_team == "away" else "away"
                # save statistics
                defender_team = take_on.home_team_name if defender_side == "home" else take_on.away_team_name
                for key, value in take_on.defenders.items():
                    if value == f"{defender_side}_{defender_num}":
                        defender_name = key
                        break
                attacking_teams.append(take_on.attacker_team)
                defending_teams.append(defender_team)
                attacking_players.append(take_on.attacker_name)
                defending_players.append(defender_name)
               
               
                valid_actions.append(
                    OneVsOne(
                        tracking_data = take_on.tracking_data_batch,
                        attacker_column_id=take_on.attacker_column_id,
                        defender_column_id=f"{defender_side}_{defender_num}",
                        attacker_side=take_on.attacking_team,
                        defender_side=defender_side,
                        is_successful=take_on.is_successful,
                    )
                )
    
    return valid_actions
    
