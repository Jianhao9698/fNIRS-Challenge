from statsmodels.graphics.tukeyplot import results
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
# from prefrontalAndExpertise import lower_bound
from visualizeData import *
from signalMapping import *
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt


signalSelected = 1

pair_total, pair_lTor, pair_rTol = corresponding_pairs()
coords_left, coords_right = generate_coords(spacing=1)

############################################################################################################
# Read CSV data
csv_file = "SPNDataChallenge0001_withHeader.csv"
df = pd.read_csv(csv_file)

#Filter for Oxy-Hb only (SIGNAL=1)
df_oxy = df[df['SIGNAL'] == signalSelected].copy()

df_oxy['TASK_MINUS_BASELINE'] = (
    df_oxy.groupby('CHANNEL')['TASK_MINUS_BASELINE']
          .transform(zscore)
)

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
    unique_sessions = subdf['SESSION'].unique()
    # print("unique_sessions: ", unique_sessions)
    try:
        session_id = int(unique_sessions[0])
        # print("session_id: ", session_id, type(session_id))
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
    # print("session_label: ", session_label)
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

    diffs = []
    for (l_ch, r_ch) in valid_pairs:
        val_left = subdf[subdf['CHANNEL'] == l_ch]['TASK_MINUS_BASELINE'].values
        val_right = subdf[subdf['CHANNEL'] == r_ch]['TASK_MINUS_BASELINE'].values

        if len(val_left) == 1 and len(val_right) == 1:
            diff = val_right[0] - val_left[0]
            diffs.append(diff)
        else:
            pass

    if len(diffs) == 0:
        continue

    lateral_diff_mean = np.mean(diffs)
    debug = False
    if debug:
        results.append({
            'SUBJECT': subj_id,
            'SESSION': session_label,
            'DIFF_MEAN': lateral_diff_mean,
            'N_PAIRS': len(diffs)
        })
    else:
        results.append({
            'SUBJECT': subj_id,
            'SESSION': session_label,
            'DIFF_MEAN': lateral_diff_mean
        })

df_diff = pd.DataFrame(results)
print(df_diff.head())

# df_oxy['TASK_MINUS_BASELINE'] = (
#     df_oxy.groupby('CHANNEL')['TASK_MINUS_BASELINE']
#           .transform(zscore)
# )
def remove_outliers(subdf):
    q1 = subdf['DIFF_MEAN'].quantile(0.25)
    q3 = subdf['DIFF_MEAN'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return subdf[(subdf['DIFF_MEAN'] >= lower) & (subdf['DIFF_MEAN'] <= upper)]

df_diff = df_diff.groupby('SESSION', group_keys=False).apply(remove_outliers)
df_diff.reset_index(drop=True, inplace=True)
########################################################################################

import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp


import pandas as pd
import numpy as np
from scipy.stats import kruskal

# Kruskal-Wallis test

groups = sessions_dict.values()
data_by_group = [df_diff[df_diff['SESSION'] == g]['DIFF_MEAN'] for g in groups]

# kruskal
h_stat, p_k = kruskal(*data_by_group)
print("===== Kruskal–Wallis Test =====")
print(f"H-statistic = {h_stat:.4f}, p-value = {p_k:.4e}")

if p_k < 0.05:
    print("Kruskal-Wallis significant，doing post hoc tests...")
    # 2) Dunn’s test
    #   scikit_posthocs.posthoc_dunn()
    df_for_posthoc = df_diff[['SESSION', 'DIFF_MEAN']].copy()
    df_for_posthoc.columns = ['group', 'value']  # rename for clarity
    posthoc_res = sp.posthoc_dunn(df_for_posthoc, val_col='value', group_col='group', p_adjust='fdr_bh')

    print("\n===== Dunn’s Posthoc with FDR correction =====")
    print(posthoc_res)
else:
    print("Kruskal-Wallis test results not significant, no post hoc tests performed.")


# 3a) U-test: Novice vs Consultant, later more
import itertools
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

groups = sessions_dict.values()
group_pairs = list(itertools.combinations(groups, 2))

p_values = []
pair_labels = []

for g1, g2 in group_pairs:
    data_g1 = df_diff[df_diff['SESSION'] == g1]['DIFF_MEAN']
    data_g2 = df_diff[df_diff['SESSION'] == g2]['DIFF_MEAN']

    u_stat, p_val = mannwhitneyu(data_g1, data_g2, alternative='two-sided')
    p_values.append(p_val)
    pair_labels.append((g1, g2))

reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
    p_values, alpha=0.05, method='fdr_bh'
)

print("===== Pairwise Mann–Whitney U with multiple corrections (FDR) =====")
for (g1, g2), p_raw, p_corr, r in zip(pair_labels, p_values, pvals_corrected, reject):
    print(f"{g1} vs {g2}: raw p={p_raw:.4e}, corrected p={p_corr:.4e}, reject={r}")
from pandas.api.types import CategoricalDtype

session_order = ['Consultant', 'Registrar', 'Novice']
cat_type = CategoricalDtype(categories=session_order, ordered=True)

# 将 SESSION 列转为有序类别
df_diff['SESSION'] = df_diff['SESSION'].astype(cat_type)

# import seaborn as sns
#
# plt.figure(figsize=(6, 4))
# sns.boxplot(data=df_diff, x='SESSION', y='DIFF_MEAN')
# sns.stripplot(data=df_diff, x='SESSION', y='DIFF_MEAN', color='red', alpha=0.5)
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# plt.title("Differences of Left / Right Lobe in Sessions (Right - Left)")
# plt.ylabel("Difference (R - L)")
# plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(7, 5))

box = sns.boxplot(
    data=df_diff,
    x='SESSION',
    y='DIFF_MEAN',
    palette="Set2",
    width=0.5,
    fliersize=4,
    dodge=False,
    boxprops=dict(alpha=0.7),
    showcaps=True
)

strip = sns.stripplot(
    data=df_diff,
    x='SESSION',
    y='DIFF_MEAN',
    color='red',
    alpha=0.6,
    jitter=True,
    dodge=False,
    size=6,
    label='Individual Data'
)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
# Adjust the visuals
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Differences of Left / Right Lobe in Sessions (Right - Left)", fontsize=14)
plt.ylabel("Difference (R - L)")
plt.xlabel("SESSION")

plt.tight_layout()
plt.show()
