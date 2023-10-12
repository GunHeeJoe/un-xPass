"""Data visualisation."""
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
from socceraction.spadl.config import field_length, field_width

def plot_action(
    action: pd.Series,
    surface=None,
    show_action=True,
    show_visible_area=True,
    ax=None,
    surface_kwargs={},
    log_bool = True,
    home_team_id = None
) -> None:
    """Plot a SPADL action with 360 freeze frame.

    Parameters
    ----------
    action : pandas.Series
        A row from the actions DataFrame.
    surface : np.arry, optional
        A surface to visualize on top of the pitch.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
    surface_kwargs : dict, optional
        Keyword arguments to pass to the surface plotting function.
    """
    # parse freeze frame
    action = play_left_to_right(action, home_team_id)

    freeze_frame = pd.DataFrame.from_records(action["freeze_frame_360"])
    visible_area = action["visible_area_360"]
    
    teammate_locs = freeze_frame[freeze_frame.teammate]
    opponent_locs = freeze_frame[~freeze_frame.teammate]

    # set up pitch
    p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)

    if ax is None:
        _, ax = p.draw(figsize=(12, 8))
    else:
        p.draw(ax=ax)

    # plot action
    if surface is not None:
        x_bin, y_bin = _get_cell_indexes(action["end_x"],action["end_y"])
        probability = surface[y_bin][x_bin]

        ax.text(x=0,y=-5,s=probability)
    
    else:
        ax.legend(['blue : teammate','red : opponent','white : start & end'],
                  loc='upper right',fontsize='x-small')

             
    if show_action:
        p.arrows(
            action["start_x"],
            action["start_y"],
            action["end_x"],
            action["end_y"],
            color="black",
            headwidth=7,
            headlength=5,
            width=2,
            ax=ax,
        )
        
    # plot visible area
    if show_visible_area:
        p.polygon([visible_area], color=(236 / 256, 236 / 256, 236 / 256, 0.5), ax=ax)
    # plot freeze frame
    p.scatter(teammate_locs.x, teammate_locs.y, c="#6CABDD", s=80, ec="k", ax=ax)
    p.scatter(opponent_locs.x, opponent_locs.y, c="#C8102E", s=80, ec="k", ax=ax)
    p.scatter(action["start_x"], action["start_y"], c="w", s=20, ec="k", ax=ax)
    p.scatter(action["end_x"], action["end_y"], c="w", s=20, ec="k", ax=ax)
    #plot surface
    if surface is not None:
        if log_bool:
            img = ax.imshow(np.log(surface), extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)
        else:
            img = ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)
        plt.colorbar(img,ax=ax)
    return ax

def _get_cell_indexes(x, y):
    y_bins, x_bins = (68, 104)
    x_bin = np.clip(x / 105 * x_bins, 0, x_bins - 1).astype(np.uint8)
    y_bin = np.clip(y / 68 * y_bins, 0, y_bins - 1).astype(np.uint8)
    return x_bin, y_bin


def play_left_to_right(actions, home_team_id):
    ltr_actions = actions.copy()
    away_idx = actions.team_id != home_team_id
    print("away_idx : ",away_idx)
    if away_idx:
        for col in ["start_x", "end_x"]:
            ltr_actions[col] = field_length - actions[col]
        for col in ["start_y", "end_y"]:
            ltr_actions[col] = field_width - actions[col]
        
        ltr_actions['freeze_frame_360'] = freeze_left_to_right(ltr_actions)
    return ltr_actions

def freeze_left_to_right(actions):
    freezedf = pd.DataFrame.from_records(actions["freeze_frame_360"])

    freezedf["x"] = field_length - freezedf["x"].values
    freezedf["y"] = field_width - freezedf["y"].values          

    freezedf_list = freezedf.to_records(index=False).tolist()
    freezedf_dict = [{
        'teammate': item[0],
        'actor': item[1],
        'keeper': item[2],
        'x': item[3],
        'y': item[4]}
    for item in freezedf_list]

    return freezedf_dict


    