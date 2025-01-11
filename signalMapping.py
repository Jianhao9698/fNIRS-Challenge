#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
We in this script to:
1) Read the SPNDataChallenge0001.csv
2) Filter rows by SIGNAL=1 (HbO), could use SIGNAL = 2 for HHb
3) For each SUBJECT, gather the 24 channels' TASK_MINUS_BASELINE
4) Plot a topographic map via 2D interpolation and contourf
5) Save or show the figure

"""
import os
from scipy.stats import zscore
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from visualizeData import generate_coords

# -----------------------
# 1) Define channel->(x,y) layout
#    The numbers below are purely illustrative.
#    Replace them with coordinates matching your device/figure if needed.

# channel_coords = {
#     1: (0.5, 4.0),
#     2: (1.5, 4.0)}

left_side_coords, right_side_coords = generate_coords(spacing=1)

# Complete channel_coords by combining left and right sides
channel_coords_left = {}
channel_coords_right = {}
for ch, xy in left_side_coords.items():
    channel_coords_left[ch] = xy

for ch, xy in right_side_coords.items():
    channel_coords_right[ch] = xy

# You can tweak these XY positions so that they visually match
# your figure's arrangement (e.g., left channels on the left, etc.).

def min_max_normalize(x: pd.Series) -> pd.Series:
    """
    Min-Max norm, to [0, 1]
    """
    return (x - x.min()) / (x.max() - x.min())

# -----------------------
def main():
    # 2) Read CSV data
    csv_file = "SPNDataChallenge0001_withHeader.csv"
    df = pd.read_csv(csv_file)

    # 3) Filter for Oxy-Hb only (SIGNAL=1)
    df_oxy = df[df['SIGNAL'] == 1].copy()
    df_oxy['TASK_MINUS_BASELINE'] = (
        df_oxy.groupby('SUBJECT')['TASK_MINUS_BASELINE']
        .transform(zscore)
    )
    # df_oxy['TASK_MINUS_BASELINE'] = df_oxy['TASK_MINUS_BASELINE'].transform(min_max_normalize)

    # df_oxy = df_oxy[ (df_oxy['TASK_MINUS_BASELINE'] > -50) &
    #                  (df_oxy['TASK_MINUS_BASELINE'] <  50) ]

    grouped_channel_counts = df_oxy.groupby(['SESSION', 'SUBJECT'])['CHANNEL'].nunique()

    avg_channel_per_session = grouped_channel_counts.groupby('SESSION').mean().reset_index()
    avg_channel_per_session.columns = ['SESSION', 'AVG_CHANNEL_COUNT']

    print(avg_channel_per_session)

    subject_ids = df_oxy['SUBJECT'].unique()
    subject_ids.sort()

    channel_counts = df_oxy.groupby('SUBJECT')['CHANNEL'].count()
    valid_subjects = channel_counts[channel_counts >= 6].index # Keep unique channels num >= 24 subjects
    df_oxy = df_oxy[df_oxy['SUBJECT'].isin(valid_subjects)]
    count_1 = 0
    count_2 = 0
    count_3 = 0
    subject_sessions = df_oxy.groupby('SUBJECT')['SESSION'].unique()
    for subj_id, sessions in subject_sessions.items():
        # print(f"Subject {subj_id} has sessions: {sessions}")
        if sessions[0] == 1:
            count_1 += 1
        elif sessions[0] == 2:
            count_2 += 1
        elif sessions[0] == 3:
            count_3 += 1
        else:
            raise ValueError(f"Unknown session {sessions[0]}")
    print("subject_sessions: ", (count_1, count_2, count_3), "channel_counts: ", channel_counts)

    print("Length after filtering: ", len(df_oxy['SUBJECT'].unique()), "Before: ", len(df['SUBJECT'].unique()))
    # For demonstration, we'll loop over each subject
    # and generate a topographic map of TASK_MINUS_BASELINE
    for subj_id in subject_ids:
        # a) extract the data for this subject
        subdf = df_oxy[df_oxy['SUBJECT'] == subj_id]

        # b) expect 24 channels in the dataset (1..24).
        #    Let's get the channel-based values:
        #    We'll create arrays for interpolation: coords and vals
        coords_list_left = []
        coords_list_right = []
        vals_list_left = []
        vals_list_right = []
        channels_list_left = []
        channels_list_right = []
        missing_coords_left = []
        missing_coords_right = []
        sessions_dict = {}
        sessions_dict = {1: 'Consultant', 2: 'Registrar', 3: 'Novice'}
        for ch in range(1, 25):  # channel 1..24
            # in case some channels are missing, check if row exists
            row_ch = subdf[subdf['CHANNEL'] == ch]
            if len(row_ch) == 1:
                # get that channel's value
                val = row_ch['TASK_MINUS_BASELINE'].values[0]
                # get the channel's (x,y) from our dict
                if ch in channel_coords_left:
                    x, y = channel_coords_left[ch]
                    coords_list_left.append([x, y])
                    vals_list_left.append(val)
                    channels_list_left.append(ch)

                elif ch in channel_coords_right:
                    x, y = channel_coords_right[ch]
                    coords_list_right.append([x, y])
                    vals_list_right.append(val)
                    channels_list_right.append(ch)
                else:
                    pass
                    # print(f"Warning: channel {ch} not found in channel_coords mapping.")
            else:
                # Could be missing data; skip or handle as needed
                # NaN or duplicated channel
                if ch in channel_coords_left:
                    missing_coords_left.append(ch)
                elif ch in channel_coords_right:
                    missing_coords_right.append(ch)
                # if ch in channel_coords_left:
                #     x, y = channel_coords_left[ch]
                #     coords_list_left.append([x, y])
                # elif ch in channel_coords_right:
                #     x, y = channel_coords_right[ch]
                #     coords_list_right.append([x, y])
                # else:
                #     raise(ValueError(f"Channel {ch} not found in channel_coords mapping."))
                # print(f"Subject {subj_id}, channel {ch} is missing or duplicated, skipping")

        if len(coords_list_left) <= 4 or len(coords_list_right) <= 4:
            # Interpolation won't work well if we have too few points
            print(f"Subject {subj_id} has insufficient valid channel data, skipping plot.")
            # print(channels_list, len(channels_list))
            coords_arr_left = np.array(coords_list_left)
            coords_arr_right = np.array(coords_list_right)
            vals_arr_left = np.array(vals_list_left)
            vals_arr_right = np.array(vals_list_right)

            # c) Create a grid for interpolation

            grid_x_l, grid_y_l = None, None
            grid_x_r, grid_y_r = None, None
            grid_z_l = None
            grid_z_r = None
        else:
            # print(channels_list, len(channels_list))
            coords_arr_left = np.array(coords_list_left)
            coords_arr_right = np.array(coords_list_right)
            vals_arr_left = np.array(vals_list_left)
            vals_arr_right = np.array(vals_list_right)

            # c) Create a grid for interpolation
            x_min_l, x_max_l = coords_arr_left[:, 0].min() - 0.5, coords_arr_left[:, 0].max() + 0.5
            y_min_l, y_max_l = coords_arr_left[:, 1].min() - 0.5, coords_arr_left[:, 1].max() + 0.5

            x_min_r, x_max_r = coords_arr_right[:, 0].min() - 0.5, coords_arr_right[:, 0].max() + 0.5
            y_min_r, y_max_r = coords_arr_right[:, 1].min() - 0.5, coords_arr_right[:, 1].max() + 0.5

            grid_size = 100
            grid_x_l, grid_y_l = np.mgrid[x_min_l:x_max_l:grid_size * 1j,
                             y_min_l:y_max_l:grid_size * 1j]
            grid_x_r, grid_y_r = np.mgrid[x_min_r:x_max_r:grid_size * 1j,
                             y_min_r:y_max_r:grid_size * 1j]
            # d) Interpolate: 'cubic' might fail if < needed points, so 'linear' is safer
            grid_z_l = griddata(coords_arr_left, vals_arr_left, (grid_x_l, grid_y_l), method='linear')
            grid_z_r = griddata(coords_arr_right, vals_arr_right, (grid_x_r, grid_y_r), method='linear')


        # e) Plot
        # plt.figure(figsize=(12, 5))
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(nrows=1, ncols=3,
                               width_ratios=[1, 1, 0.1],  # 分配三列的宽度
                               wspace=0.1)  # 子图之间的水平间距

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_cb = fig.add_subplot(gs[0, 2])  # 这一列专门用来放 colorbar
        ax_left.set_xlim(-0.1, 2.1)
        ax_left.set_ylim(-2.1, 0.1)
        ax_right.set_xlim(-0.1, 2.1)
        ax_right.set_ylim(-2.1, 0.1)
        ax_left.set_aspect('equal', 'box')
        ax_right.set_aspect('equal', 'box')
        # plt_is_left = [channels_list[i] in channel_coords_left for i in range(len(channels_list))]
        # left_indices = [i for i, x in enumerate(plt_is_left) if x]
        # right_indices = [i for i, x in enumerate(plt_is_left) if not x]
        # print(plt_is_left, left_indices)

        # Mark channel positions
        # plt.subplot(1, 2, 1)
        # contour/contourf
        # If grid_z has NaNs, we can mask them out or set levels='auto'
        # cs_l = ax_left.contourf(X, Y, Z1, cmap='bwr', levels=20, vmin=-1, vmax=1)
        # print(df_oxy[df_oxy['SUBJECT'] == subj_id])
        # print(df_oxy[df_oxy['SUBJECT'] == subj_id].shape[0])
        for num in range(df_oxy[df_oxy['SUBJECT'] == subj_id].shape[0]):
            try:
                ax_left.set_title(f"Subject {subj_id} Right lobe, {sessions_dict[df_oxy[df_oxy['SUBJECT'] == subj_id].iloc[num, 1]]}")
                break
            except:
                continue
        if grid_z_l is not None:
            cs_l = ax_left.contourf(grid_x_l, grid_y_l, grid_z_l, levels=20, vmin=0, vmax=1, cmap='bwr', alpha=0.8)
        else:
            cs_l = None
        # plt.colorbar(cs_l, label='TASK_MINUS_BASELINE')
        for i, val in enumerate(coords_list_left):
            ch_x = coords_arr_left[i, 0]
            ch_y = coords_arr_left[i, 1]
            ax_left.plot(ch_x, ch_y, 'ko', markersize=2)
            txt = f'{channels_list_left[i]}'
            ax_left.text(
                ch_x, ch_y,
                txt,
                color='black',
                ha='center', va='center',
                fontweight='bold',
                fontsize=8,
                bbox=dict(
                    facecolor='yellow',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
        for i, missing in enumerate(missing_coords_left):
            # print(i, missing, type(missing))
            # missing = missing - 1
            ch_x = channel_coords_left[missing][0]
            ch_y = channel_coords_left[missing][1]
            ax_left.plot(ch_x, ch_y, 'ko', markersize=2)
            txt = f'{missing}'
            ax_left.text(
                ch_x, ch_y,
                txt,
                color='black',
                ha='center', va='center',
                fontweight='bold',
                fontsize=8,
                bbox=dict(
                    facecolor='yellow',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
        ax_left.axis('off')

        # plt.subplot(1, 2, 2)
        for num in range(df_oxy[df_oxy['SUBJECT'] == subj_id].shape[0]):
            try:
                ax_right.set_title(f"Subject {subj_id} Left lobe, {sessions_dict[df_oxy[df_oxy['SUBJECT'] == subj_id].iloc[num, 1]]}")
                break
            except:
                continue

        if grid_z_l is not None:
            cs_r = ax_right.contourf(grid_x_r, grid_y_r, grid_z_r, levels=128, vmin=0, vmax=1, cmap='bwr', alpha=0.8)
        else:
            cs_r = None
        # cbar = plt.colorbar(cs_r, label='TASK_MINUS_BASELINE')
        # cs_r.set_clim(0, 1)
        # cbar.update_normal(cs_r)
        # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        for i, val in enumerate(coords_list_right):
            ch_x = coords_arr_right[i, 0]
            ch_y = coords_arr_right[i, 1]
            ax_right.plot(ch_x, ch_y, 'ko', markersize=2)
            txt = f'{channels_list_right[i]}'
            ax_right.text(
                ch_x, ch_y,
                txt,
                color='black',
                ha='center', va='center',
                fontweight='bold',
                fontsize=8,
                bbox=dict(
                    facecolor='yellow',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
        for i, missing in enumerate(missing_coords_right):
            # print(i, missing, type(missing))
            # missing = missing - 1
            ch_x = channel_coords_right[missing][0]
            ch_y = channel_coords_right[missing][1]
            ax_right.plot(ch_x, ch_y, 'ko', markersize=2)
            txt = f'{missing}'
            ax_right.text(
                ch_x, ch_y,
                txt,
                color='black',
                ha='center', va='center',
                fontweight='bold',
                fontsize=8,
                bbox=dict(
                    facecolor='yellow',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
        ax_right.axis('off')

        if cs_r is not None:
            cbar = plt.colorbar(cs_r, cax=ax_cb)
            cbar.set_label("TASK_MINUS_BASELINE")
            cs_r.set_clim(0, 1)
            cbar.update_normal(cs_r)
        else:
            pass

        output_folder = "./topo_maps"
        os.mkdir(output_folder) if not os.path.exists(output_folder) else None
        outname = f"{output_folder}/Subject_{subj_id}_topo.png"
        plt.savefig(outname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved topographic map for subject {subj_id} to {outname}")

    print("All subject plots generated.")


# -----------------------
if __name__ == "__main__":
    main()
