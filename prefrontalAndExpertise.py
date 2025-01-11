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

signalSelected = 2  # 1 for Oxy-Hb, 2 for Deoxy-Hb

############################################################################################################
# 2) Read CSV data
csv_file = "SPNDataChallenge0001_withHeader.csv"
df = pd.read_csv(csv_file)

# 3) Filter for Oxy-Hb only (SIGNAL=1)
df_oxy = df[df['SIGNAL'] == signalSelected].copy()

def min_max_normalize(x):
    if x.max() == x.min():
        return x
    return (x - x.min()) / (x.max() - x.min())

# df_oxy['TASK_MINUS_BASELINE'] = (
#     df_oxy.groupby('CHANNEL')['TASK_MINUS_BASELINE']
#           .transform(min_max_normalize)
# )
df_oxy['TASK_MINUS_BASELINE'] = (
    df_oxy.groupby('CHANNEL')['TASK_MINUS_BASELINE']
          .transform(zscore)
)

# 4) List all unique SUBJECT IDs
subject_ids = df_oxy['SUBJECT'].unique()
subject_ids.sort()

channel_counts = df_oxy.groupby('SUBJECT')['CHANNEL'].nunique()
valid_subjects = channel_counts[channel_counts >= 6].index  # Keep unique channels num >= 6
df_oxy = df_oxy[df_oxy['SUBJECT'].isin(valid_subjects)]
print("Length after filtering: ", len(df_oxy['SUBJECT'].unique()),
      "Before: ", len(df['SUBJECT'].unique()))

sessions_dict = {1: 'Consultant',
                 2: 'Registrar',
                 3: 'Novice'}
results = []


def get_session_name(subdf):
    # print(subdf, subdf['SESSION'])
    unique_sessions = subdf['SESSION'].unique()
    print("unique_sessions: ", unique_sessions)
    try:
        session_id = int(unique_sessions[0])
        print("session_id: ", session_id, type(session_id))
        return sessions_dict[session_id]
    except:
        return None


for subj_id in subject_ids:
    subdf = df_oxy[df_oxy['SUBJECT'] == subj_id].copy()
    if len(subdf) == 0:
        continue

for subj_id in subject_ids:
    # a) extract the data for this subject
    subdf = df_oxy[df_oxy['SUBJECT'] == subj_id]
    session_label = get_session_name(subdf)
    print("session_label: ", session_label)
    if session_label is not None:
        print(f"Processing Subject {subj_id} ({session_label})")
    else:
        # print("subdf here: ", subdf)
        continue
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

    valid_pairs = []
    for left_ch in pair_lTor.keys():
        right_ch = pair_lTor[left_ch]
        if (left_ch in left_channels_with_data) and (right_ch in right_channels_with_data):
            valid_pairs.append((left_ch, right_ch))

    means = []
    for (l_ch, r_ch) in valid_pairs:
        val_left = subdf[subdf['CHANNEL'] == l_ch]['TASK_MINUS_BASELINE'].values
        val_right = subdf[subdf['CHANNEL'] == r_ch]['TASK_MINUS_BASELINE'].values

        if len(val_left) == 1 and len(val_right) == 1:
            sum = val_right[0] + val_left[0]
            means.append(sum)
        else:
            pass

    if len(means) == 0:
        continue

    forehead_means_mean = np.mean(means)

    results.append({
        'SUBJECT': subj_id,
        'SESSION': session_label,
        'Forehead_MEAN': forehead_means_mean
    })

def remove_outliers(subdf):
    q1 = subdf['Forehead_MEAN'].quantile(0.25)
    q3 = subdf['Forehead_MEAN'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return subdf[(subdf['Forehead_MEAN'] >= lower) & (subdf['Forehead_MEAN'] <= upper)]

df_forehead = pd.DataFrame(results)
df_forehead = df_forehead.groupby('SESSION', group_keys=False).apply(remove_outliers)
df_forehead.reset_index(drop=True, inplace=True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy import stats

k_values = [1, 2, 3, 4, 5, 6, 7, 8]
results_k = []

for subj_id in subject_ids:
    subdf = df_oxy[df_oxy['SUBJECT'] == subj_id].copy()
    if len(subdf) == 0:
        continue

    unique_sessions = subdf['SESSION'].unique()
    if len(unique_sessions) == 0:
        continue
    session_id = int(unique_sessions[0])
    if session_id not in sessions_dict:
        continue
    session_label = sessions_dict[session_id]


    channel_vals = subdf[['CHANNEL', 'TASK_MINUS_BASELINE']].dropna()

    num_channels = len(channel_vals)
    if num_channels < 1:
        continue

    channel_vals_sorted = channel_vals.sort_values(
        by='TASK_MINUS_BASELINE', ascending=False
    )

    for k in k_values:
        if num_channels < k:
            continue

        top_k_vals = channel_vals_sorted['TASK_MINUS_BASELINE'].iloc[:k]
        top_k_mean = top_k_vals.mean()

        results_k.append({
            'SUBJECT': subj_id,
            'SESSION': session_label,
            'K': k,
            'TOPK_MEAN': top_k_mean
        })

df_topk = pd.DataFrame(results_k)
def remove_outliers(subdf):
    q1 = subdf['TOPK_MEAN'].quantile(0.25)
    q3 = subdf['TOPK_MEAN'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return subdf[(subdf['TOPK_MEAN'] >= lower) & (subdf['TOPK_MEAN'] <= upper)]
df_topk = df_topk.groupby('SESSION', group_keys=False).apply(remove_outliers)
df_topk.reset_index(drop=True, inplace=True)
print(df_topk.head(10))


sns.lineplot(
    data=df_topk,
    x='K',
    y='TOPK_MEAN',
    hue='SESSION',
    marker='o'
)
plt.title("Top-K Activation Comparison")
plt.ylabel("Mean of Top-K Activation")
plt.show()
# print(df_forehead.head())

########################################################################################
import scikit_posthocs as sp

for k_val in k_values:
    df_k = df_topk[df_topk['K'] == k_val]
    groups = df_k.groupby('SESSION')['TOPK_MEAN'].apply(list)

    if len(groups) <= 1:
        print(f"[K={k_val}] Not enough groups to compare.")
        continue

    # Kruskal-Wallis test
    h_stat, p_val = stats.kruskal(*groups)
    print(f"[K={k_val}] Kruskal-Wallis H ={h_stat:.3f}, p={p_val:.4e}")

    # Consider post-hoc test if p-value is significant
    if p_val < 0.05 and len(groups) > 2:
        # Compute Dunn's test for multiple comparisons
        # scikit-posthocs 的 posthoc_dunn()
        posthoc_data = df_k[['SESSION', 'TOPK_MEAN']]
        posthoc_res = sp.posthoc_dunn(
            posthoc_data,
            val_col='TOPK_MEAN',
            group_col='SESSION',
            p_adjust='bonferroni'  # Use Bonferroni correction
        )
        print(f"    Dunn's test (p value matrix, K={k_val}):\n{posthoc_res}\n")

    else:
        print(f"    [K={k_val}] not significant, no post-hoc test performed.")

import seaborn as sns

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_forehead, x='SESSION', y='Forehead_MEAN')
sns.stripplot(data=df_forehead, x='SESSION', y='Forehead_MEAN', color='red', alpha=0.5)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Forehead activation in different Sessions (Total)")
plt.ylabel("Mean Value")
plt.show()
