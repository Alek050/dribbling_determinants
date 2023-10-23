import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import matplotlib.animation as animation
from tqdm import tqdm

def plot_pitch(
    field_dimen:tuple=(106.0, 68.0), field_color:str="green", linewidth:float=2, markersize:float=20
) -> Tuple[plt.figure, plt.axis]:
    """Plots a soccer pitch. All distance units converted to meters.
    Based on: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Viz.py

    Args:
        field_dimen (tuple, optional): (length, width) of field in meters. Default is (106,68). Defaults to (106.0, 68.0).
        field_color (str, optional): color of field. options are {'green','white'}. Defaults to "green".
        linewidth (float, optional): width of lines. Defaults to 2.
        markersize (float, optional): size of markers (e.g. penalty spot, centre spot, posts). Defaults to 20.

    Returns:
        Tuple[plt.figure, plt.axis]: figure and aixs objects
    """
    fig, ax = plt.subplots(figsize=(19.2, 10.8))  # create a figure
    # decide what color we want the field to be. Default is green,
    # but can also choose white
    if field_color == "green":
        ax.set_facecolor("mediumseagreen")
        lc = "whitesmoke"  # line color
        pc = "w"  # 'spot' colors
    elif field_color == "white":
        lc = "k"
        pc = "k"
    # ALL DIMENSIONS IN m
    border_dimen = (3, 3)  # include a border arround of the field of width 3m
    meters_per_yard = 0.9144  # unit conversion from yards to meters
    half_pitch_length = field_dimen[0] / 2.0  # length of half pitch
    half_pitch_width = field_dimen[1] / 2.0  # width of half pitch
    signs = [-1, 1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8 * meters_per_yard
    box_width = 20 * meters_per_yard
    box_length = 6 * meters_per_yard
    area_width = 44 * meters_per_yard
    area_length = 18 * meters_per_yard
    penalty_spot = 12 * meters_per_yard
    corner_radius = 1 * meters_per_yard
    D_length = 8 * meters_per_yard
    D_radius = 10 * meters_per_yard
    D_pos = 12 * meters_per_yard
    centre_circle_radius = 10 * meters_per_yard
    # plot half way line # center circle
    ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=linewidth, zorder=-1)
    ax.scatter(0.0, 0.0, marker="o", facecolor=lc, linewidth=0, s=markersize, zorder=-1)
    y = np.linspace(-1, 1, 50) * centre_circle_radius
    x = np.sqrt(centre_circle_radius**2 - y**2)
    ax.plot(x, y, lc, linewidth=linewidth, zorder=-1)
    ax.plot(-x, y, lc, linewidth=linewidth, zorder=-1)
    for s in signs:  # plots each line seperately
        # plot pitch boundary
        ax.plot(
            [-half_pitch_length, half_pitch_length],
            [s * half_pitch_width, s * half_pitch_width],
            lc,
            linewidth=linewidth, zorder=-1
            
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-half_pitch_width, half_pitch_width],
            lc,
            linewidth=linewidth, zorder=-1
        )
        # goal posts & line
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length],
            [-goal_line_width / 2.0, goal_line_width / 2.0],
            pc + "s",
            markersize=6 * markersize / 20.0,
            linewidth=linewidth, zorder=-1
        )
        # 6 yard box
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth, zorder=-1
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * box_length],
            [-box_width / 2.0, -box_width / 2.0],
            lc,
            linewidth=linewidth, zorder=-1
        )
        ax.plot(
            [
                s * half_pitch_length - s * box_length,
                s * half_pitch_length - s * box_length,
            ],
            [-box_width / 2.0, box_width / 2.0],
            lc,
            linewidth=linewidth, zorder=-1
        )
        # penalty area
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth, zorder=-1
        )
        ax.plot(
            [s * half_pitch_length, s * half_pitch_length - s * area_length],
            [-area_width / 2.0, -area_width / 2.0],
            lc,
            linewidth=linewidth, zorder=-1
        )
        ax.plot(
            [
                s * half_pitch_length - s * area_length,
                s * half_pitch_length - s * area_length,
            ],
            [-area_width / 2.0, area_width / 2.0],
            lc,
            linewidth=linewidth, zorder=-1
        )
        # penalty spot
        ax.scatter(
            s * half_pitch_length - s * penalty_spot,
            0.0,
            marker="o",
            facecolor=lc,
            linewidth=0,
            s=markersize, zorder=-1
        )
        # corner flags
        y = np.linspace(0, 1, 50) * corner_radius
        x = np.sqrt(corner_radius**2 - y**2)
        ax.plot(
            s * half_pitch_length - s * x, -half_pitch_width + y, lc, linewidth=linewidth, zorder=-1
        )
        ax.plot(
            s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=linewidth, zorder=-1
        )
        # draw the D
        y = (
            np.linspace(-1, 1, 50) * D_length
        )  # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2 - y**2) + D_pos
        ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=linewidth, zorder=-1)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0] / 2.0 + border_dimen[0]
    ymax = field_dimen[1] / 2.0 + border_dimen[1]
    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-ymax, ymax])
    ax.set_axisbelow(True)
    return fig, ax

def save_match_clip(
    *,
    df: pd.DataFrame,
    home_team_name:str,
    away_team_name:str,
    fpath:str="data/clips",
    fname:str="clip_test",
    figax:plt.axes=None,
    team_colors:tuple=("g", "b"),
    alpha=0.7,
    frame_rate:int=25,
    verbose=True,
):
    """Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with
    name 'fname'
    Based on:  Source: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_Viz.py

    Args:
        df (pd.DataFrame): (partial) tracking data of the match, from which a clip must be made.
        fpath (str, optional): directory to save the clip. Defaults to "data/clips".
        fname (str, optional): name of the saved clip. Defaults to "clip_test".
        figax (plt.axes, optional): fixaxes to which to plot the tracking data, if none,
            the plot_pitch function will be used. Defaults to None.
        team_colors (tuple, optional): colors for home and away team. Defaults to ("g", "b").
        alpha (float, optional): alpha for the player dots. Defaults to 0.7.
        frame_rate (int, optional): frame rate of the movie. Defaults to 25.
        verbose (bool, optional): Whether to print info about progress in the terminal. Defaults to True.
    """

    ht_columns = [x for x in df.columns if "home" in x]
    at_columns = [x for x in df.columns if "away" in x]

    hometeam = df.loc[:, ht_columns]
    awayteam = df.loc[:, at_columns]

    # check that indices match first
    assert np.all(
        hometeam.index == awayteam.index
    ), "Home and away team Dataframe indices must be the same"

    index = hometeam.index

    metadata = dict(title="Tracking Data", artist="Matplotlib")
    writer = animation.FFMpegWriter(fps=frame_rate, metadata=metadata)
    fname = fpath + "/" + fname + ".mp4"  # path and filename

    # create football pitch

    if figax is None:
        fig, ax = plot_pitch()
    else:
        fig, ax = figax
    fig.set_tight_layout(True)

    # set match name
    ax.text(
        106 / -2.0 + 2,
        68 / 2.0 + 1.0,
        home_team_name,
        fontsize=14,
        color=team_colors[0],
    )
    ax.text(
        106 / 2 - 15,
        68 / 2.0 + 1.0,
        away_team_name,
        fontsize=14,
        color=team_colors[1],
    )

    # Generate movie
    with writer.saving(fig, fname, 100):
        if verbose:
            index = tqdm(index, desc="Createing clip ", unit=" frames")
        for i in index:
            figobjs = []
            for team, c in zip([hometeam.loc[i], awayteam.loc[i]], team_colors):
                x_columns = [x for x in team.keys() if x[-2:] == "_x" and c != "ball_x"]
                y_columns = [y for y in team.keys() if y[-2:] == "_y" and c != "ball_y"]
                objs = ax.scatter(
                    team[x_columns], team[y_columns], c=c, alpha=alpha, s=190, zorder=1.
                )
                figobjs.append(objs)
                for x, y in zip(x_columns, y_columns):
                    if pd.isnull(team[x]):
                        continue
                    corr = 0.2 if len(x.split("_")[1]) == 1 else 0.5
                    objs = ax.text(
                        team[x] - corr,
                        team[y] - 0.2,
                        x.split("_")[1],
                        fontsize=9,
                        c="white",
                        zorder=2.,
                    )
                    figobjs.append(objs)

            # plot ball
            objs = ax.scatter(df.loc[i, "ball_x"], df.loc[i, "ball_y"], c="black", s=80, zorder=3.)
            figobjs.append(objs)

            # include match time at the top
            time = df.loc[i, "matchtime_td"]

            objs = ax.text(
                -20.5,
                68 / 2.0 + 1,
                time,
                fontsize=14,
            )
            figobjs.append(objs)

            writer.grab_frame()

            # Delete all axis objects (except pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()

    plt.clf()
    plt.close(fig)
