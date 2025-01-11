from statsmodels.graphics.tukeyplot import results
from scipy.stats import zscore
from visualizeData import *
from signalMapping import *
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

pair_total, pair_lTor, pair_rTol = corresponding_pairs()
coords_left, coords_right = generate_coords(spacing=1)
# selected_channels_orbital = [1, 3, 14, 17]
# selected_channels_lateral = [5, 10, 15, 20]

# selected_channels_orbital = [1, 2, 13, 14]
# selected_channels_lateral = [3, 6, 17, 19]

selected_channels_orbital = [1, 14]
selected_channels_lateral = [5, 10, 7, 15, 18, 20, 12, 11, 24, 23]
############################################################################################################
# 2) Read CSV data
csv_file = "SPNDataChallenge0001_withHeader.csv"
df = pd.read_csv(csv_file)

# 3) Filter for Oxy-Hb only (SIGNAL=1)
df_oxy = df[df['SIGNAL'] == 1].copy()
df_oxy = df_oxy[df_oxy['SESSION'] == 3]
df_oxy['TASK_MINUS_BASELINE'] = (
    df_oxy.groupby('SUBJECT')['TASK_MINUS_BASELINE']
    .transform(zscore)
)
# df_oxy['TASK_MINUS_BASELINE'] = df_oxy['TASK_MINUS_BASELINE'].transform(min_max_normalize)

# df_oxy = df_oxy[ (df_oxy['TASK_MINUS_BASELINE'] > -50) &
#                  (df_oxy['TASK_MINUS_BASELINE'] <  50) ]

# 4) List all unique SUBJECT IDs
subject_ids = df_oxy['SUBJECT'].unique()
subject_ids.sort()

channel_counts = df_oxy.groupby('SUBJECT')['CHANNEL'].nunique()
valid_subjects = channel_counts[channel_counts >= 6].index # Keep unique channels num >= 24 subjects
df_oxy = df_oxy[df_oxy['SUBJECT'].isin(valid_subjects)]
print("Length after filtering: ", len(df_oxy['SUBJECT'].unique()), "Before: ", len(df['SUBJECT'].unique()))
# For demonstration, we'll loop over each subject
# and generate a topographic map of TASK_MINUS_BASELINE
# sessions_dict = {1: 'Consultant',
#                  2: 'Registrar',
#                  3: 'Novice'}
results = []

for subj_id in subject_ids:
    subdf = df_oxy[df_oxy['SUBJECT'] == subj_id].copy()
    if len(subdf) == 0:
        continue

for subj_id in subject_ids:
    # a) extract the data for this subject
    subdf = df_oxy[df_oxy['SUBJECT'] == subj_id]
    # session_label = get_session_name(subdf)
    # print("session_label: ", session_label)
    # if session_label is not None:
    #     print(f"Processing Subject {subj_id} ({session_label})")
    # else:
    #     # print("subdf here: ", subdf)
    #     continue
    # b) We expect 24 channels in the dataset (1..24).
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

############################################################################################################

    left_channels_with_data = set(subdf['CHANNEL']).intersection(set(pair_lTor.keys()))
    right_channels_with_data = set(subdf['CHANNEL']).intersection(set(pair_lTor.values()))

    # print("Left channels with data: ", left_channels_with_data)
    valid_pairs = []
    for left_ch in pair_lTor.keys():
        right_ch = pair_lTor[left_ch]
        if (left_ch in left_channels_with_data) and (right_ch in right_channels_with_data):
            if (left_ch in selected_channels_orbital) or (right_ch in selected_channels_lateral):
                valid_pairs.append((left_ch, right_ch))
    # print("Pairs selected: ", len(valid_pairs))
    orbital_means = []
    lateral_means = []
    for (l_ch, r_ch) in valid_pairs:
        val_left = subdf[subdf['CHANNEL'] == l_ch]['TASK_MINUS_BASELINE'].values
        val_right = subdf[subdf['CHANNEL'] == r_ch]['TASK_MINUS_BASELINE'].values
        # print(f"Subject {subj_id}, channels {l_ch} and {r_ch}: {val_left}, {val_right}")
        if l_ch in selected_channels_orbital:
            if len(val_left) == 1 and len(val_right) == 1:
                mean = (val_right[0] + val_left[0]) / 2
                orbital_means.append(mean)
            else:
                pass

        elif l_ch in selected_channels_lateral:
            if len(val_left) == 1 and len(val_right) == 1:
                mean = (val_right[0] + val_left[0]) / 2
                lateral_means.append(mean)
            else:
                pass

        else:
            raise ValueError("Invalid channel number")

    diff_o_minus_l = np.mean(orbital_means) - np.mean(lateral_means)

    results.append({
        'SUBJECT': subj_id,
        'DIFF_MEAN': diff_o_minus_l
    })

df_diff = pd.DataFrame(results)

########################################################################################


# t_stat, p_val = stats.ttest_1samp(df_diff['DIFF_MEAN'], popmean=0)
# print("===== 单样本 t 检验 (是否整体偏离 0) =====")
# print(f"T-stat: {t_stat:.4f}, p-value: {p_val:.4e}")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# df_diff['diff'] = df_diff['orbital_mean'] - df_diff['DIFF_MEAN']

df_diff = df_diff.dropna()
print("Length of records: ", len(df_diff), df_diff)
#  Wilcoxon test
stat, p_value = wilcoxon(df_diff['DIFF_MEAN'], alternative='greater')
print(f"Wilcoxon Signed-Rank statistic = {stat}, p-value = {p_value}")
if p_value < 0.05:
    print("Significant result: we can say that orbital activity is stronger than lateral activity.")
else:
    print("No significant result: we cannot say that orbital activity is stronger than lateral activity.")

import numpy as np
import pandas as pd
from scipy import stats


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(6, 4))

box = sns.boxplot(
    data=df_diff,
    y='DIFF_MEAN',
    color="#8EBEC7",
    width=0.3,
    fliersize=4,
    boxprops=dict(alpha=0.6),
)
box.set_xlabel("")
box.set_ylabel("Difference", fontsize=12)

strip = sns.stripplot(
    data=df_diff,
    y='DIFF_MEAN',
    color='red',
    alpha=0.7,
    size=6,
    jitter=True,
)

plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.title("Differences (Orbital - Lateral), Novice", fontsize=14)
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()
