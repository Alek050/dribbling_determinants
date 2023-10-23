import pandas as pd
import numpy as np
import subprocess
import os
from core.visualise import save_match_clip
import ffmpeg

def make_clips_and_label_sheet(
        actions: list, 
        full_match_video_loc: str, 
        save_loc: str, 
        extra_video_time: float=6.5, 
        use_video:bool =True, 
        use_td:bool = True, 
        add_label_sheet:bool = True
    ) -> None:
    """Function that creates clips of a specific action in a match. The clips are saved in the save_loc folder.
    It also adds a video clip of the tracking data and a xls file in which the user can label the clips.

    Args:
        actions (list): A list of events that should be clipped. 
        full_match_video_loc (str): The location of the full match video. Note that
            this folder should also containt a xls file called 
            `start_seconds_of_videos.xlsx` that contains the name of a match and the
            second of the video where the match really starts in the column `start_s`.
        save_loc (str): The location where the clips should be saved.
        extra_video_time (float, optional): seconds to add before and after the action 
            to get a window around the action. Defaults to 6.5.
        use_video (bool, optional): Whether to add video data of action to the clip. 
            Defaults to True.
        use_td (bool, optional): Whether to add tracking data video of the action to
            the clip. Defaults to True.
        add_label_sheet (bool, optional): Whether to create a sheet in which the user
            can label the validity and successfulness of the action. Defaults to True.
    """

    assert all([actions[x].match_name == actions[0].match_name for x in range(len(actions))]), "All actions should be from the same match"

    if add_label_sheet:
        res = {
            "name": [], 
            "datetime": [],
            "take_on_id": [], 
            "attacking_player": [], 
            "attacking_player_num": [], 
            "attacking_team_name": [],
            "defending_player": [], 
            "defending_player_num": [], 
            "valid_action":[], 
            "succesfulness": [],
            }

    match_name = actions[0].match_name

    if use_video:
        match_starters = pd.read_excel(f"{full_match_video_loc}/start_seconds_of_videos.xlsx")

        date = match_name.split(" ")[-2].replace("-", "")
        search_match_name = f"{date} {actions[0].home_team_name} vs {actions[0].away_team_name}"
        available_matches = match_starters["name"].to_list()
        overlapping_matches = [x for x in available_matches if x.startswith(search_match_name)]
        if len(overlapping_matches) == 1:
            video_name = overlapping_matches[0]
            match_start = match_starters.loc[match_starters["name"]==video_name, "start_s"].iloc[0]
        else: 
            print(f"Could not find the match start of match {match_name}")
            return

    
    clips = []
    for i, action in enumerate(actions):
        if len(action.defenders) == 0:
            continue
       
        # save info about action in dictionary to label manually later on
        if add_label_sheet:
            for def_name, def_col in action.defenders.items():
                datetime = action.tracking_data_batch[action.tracking_data_batch["databallpy_event"]=="dribble"].iloc[0]["datetime"].tz_localize(None)
                res["name"].append(match_name)
                res["datetime"].append(datetime)
                res["take_on_id"].append(i)
                res["attacking_player"].append(action.attacker_name)
                res["attacking_player_num"].append(int(action.attacker_column_id.split("_")[1]))
                res["attacking_team_name"].append(action.attacker_team)
                res["defending_player"].append(def_name)
                res["defending_player_num"].append(int(def_col.split("_")[1]))
                res["valid_action"].append(np.nan)
                res["succesfulness"].append(np.nan)
        
        if use_video:
            # add information about the event
            def_numbers = [int(x.split("_")[1]) for x in action.defenders.values()]
            information = f"Take on ID {i}, Attacking player: {action.attacker_name}, team: {action.attacking_team}, number: {action.attacker_column_id[5:]}, Defender(s): {str(def_numbers)[1:-1]}"
           
            # add the event itself
            start_s = action.start_second
            start_s += (match_start - extra_video_time)
            end_s = start_s + (2*extra_video_time)

            # make the clip a little longer
            action_clip = ffmpeg.input(f"{full_match_video_loc}/{video_name}.mp4", ss=start_s, t=end_s-start_s)
            action_clip = action_clip.filter("drawtext", text=information, fontsize=20, fontcolor="black", x="(w-text_w)/2", y="h-text_h-10", box=1)
            clips.append(action_clip)

            if use_td:
                save_match_clip(
                    df=action.tracking_data_batch, 
                    home_team_name=action.home_team_name, 
                    away_team_name=action.away_team_name,
                    fpath="data/clips", 
                    fname=f"1vs1clip_temp_{i:02d}", 
                    frame_rate=action.frame_rate,
                    verbose=False
                    )
                td_vid = ffmpeg.input(f"data/clips/1vs1clip_temp_{i:02d}.mp4")
                td_vid = td_vid.filter("drawtext", text=information, fontsize=20, fontcolor="black", x="(w-text_w)/2", y="h-text_h-10", box=1)
                clips.append(td_vid)

    if use_video:
        # output the final clip
        output_path = f"{save_loc}/{match_name}_1vs1_clips.mp4"
        output_mp4 = ffmpeg.concat(*clips, v=1, a=0).output(output_path, threads=16)
        output_mp4 = output_mp4.global_args('-y', '-progress', '-loglevel')
        output_mp4.run()
    
    # remove the temporary files
    temp_files = [x for x in os.listdir("data/clips") if x.startswith("1vs1clip_temp")]
    for file in temp_files:
        os.remove(f"data/clips/{file}")

    if add_label_sheet:
        res_df = pd.DataFrame(res)
        writer = pd.ExcelWriter(f"{save_loc}/{match_name}_1vs1_labels.xlsx", engine='xlsxwriter') 
        res_df.to_excel(writer, sheet_name='Labeling', index=False)

        # Auto-adjust columns' width
        for column in res_df.columns:
            column_width = max(res_df[column].astype(str).map(len).max(), len(column))
            col_idx = res_df.columns.get_loc(column)
            writer.sheets['Labeling'].set_column(col_idx, col_idx, column_width)

        writer._save()
