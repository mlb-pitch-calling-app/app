import pandas as pd
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os
import joblib
from datetime import datetime
from constants import (
    columns_to_keep,
    platoon_state_mapping,
    count_conditions,
    count_values,
    pitch_group_mapping,
    side_buckets,
    height_buckets,
    num_clusters,
    numerical_features,
    cluster_features,
    model_dir,
    model_keys,
    pseudo_sample_size
)

def clean_data(pitches_df, ids_df):
    pitches_df = pitches_df[columns_to_keep]
    pitches_df = pitches_df.dropna(subset=['release_speed', 'ax', 'az', 'plate_x', 'plate_z'])

    pitches_df['game_date'] = pd.to_datetime(pitches_df['game_date'], errors='coerce')
    pitches_df = pitches_df.dropna(subset=['game_date'])
    pitches_df = pitches_df[pitches_df['game_type'] == 'R']
    pitches_df = pitches_df[(pitches_df['balls'] <= 3) & (pitches_df['strikes'] <= 2)]
    pitches_df.sort_values(by=['game_date', 'game_pk', 'at_bat_number', 'pitch_number'], inplace=True)

    ids_df.rename(columns={'MLBID': 'batter', 'PLAYERNAME': 'batter_name'}, inplace=True)
    pitches_df = pitches_df.merge(ids_df[['batter', 'batter_name']], on='batter', how='left')
    pitches_df['player_name'] = pitches_df['player_name'].str.replace(r'([^,]+),\s*(.+)', r'\2 \1', regex=True)

    pitches_df['PA_ID'] = pitches_df.groupby(['game_pk', 'inning', 'inning_topbot', 'at_bat_number']).ngroup()
    pitches_df.loc[(pitches_df["pitcher"] == 694973) & (pitches_df["pitch_type"] == "SI"),"pitch_type"] = "FS"

    return pitches_df

def encode_vars(pitches_df):
    pitches_df['year'] = pitches_df['game_date'].dt.year

    pitches_df['PlatoonStateEncoded'] = pitches_df[['p_throws', 'stand']].apply(
        lambda x: platoon_state_mapping.get((x['p_throws'], x['stand'])), axis=1
    )

    pitches_df['CountEncoded'] = np.select(
        [pitches_df[['balls', 'strikes']].apply(tuple, axis=1).isin(condition) for condition in count_conditions],
        count_values,
        default=np.nan
    )
    
    pitches_df[(pitches_df['pitch_type'] != 'FA') & (pitches_df['pitch_type'] != 'PO')]
    pitches_df['PitchGroupEncoded'] = pitches_df['pitch_type'].map(pitch_group_mapping)

    return pitches_df


def add_fastball_specs(pitches_df):    
    common_pitch_type = (
        pitches_df[pitches_df['PitchGroupEncoded'] == 0]
        .groupby(['pitcher', 'stand', 'year'])['pitch_type']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={'pitch_type': 'MostCommonPitchType'})
    )

    pitches_df = pitches_df.merge(common_pitch_type, on=['pitcher', 'stand', 'year'], how='left')

    subset_df = pitches_df[pitches_df['pitch_type'] == pitches_df['MostCommonPitchType']]

    medians = (
        subset_df
        .groupby(['pitcher', 'stand', 'year'])
        [['release_speed', 'ax', 'az', 'release_pos_z', 'release_pos_x']]
        .median()
        .reset_index()
        .rename(columns={
            'release_speed': 'avg_fb_RelSpeed',
            'ax': 'avg_fb_ax0',
            'az': 'avg_fb_az0',
            'release_pos_z': 'avg_fb_RelHeight',
            'release_pos_x': 'avg_fb_RelSide'
        })
    )

    pitches_df = pitches_df.merge(medians, on=['pitcher', 'stand', 'year'], how='left')

    return pitches_df


def bucketize_plate_locations(pitches_df):
    pitches_df['PlateLocSideBucket'] = pitches_df['plate_x'].apply(lambda x: side_buckets[np.argmin(np.abs(side_buckets - x))])
    pitches_df['PlateLocHeightBucket'] = pitches_df['plate_z'].apply(lambda x: height_buckets[np.argmin(np.abs(height_buckets - x))])

    return pitches_df


def scale_data(pitches_df, save=False):
    pitches_df['BatterLefty'] = (pitches_df['stand'] == 'L').astype(int)
    pitches_df['PitcherLefty'] = (pitches_df['p_throws'] == 'L').astype(int)

    pitches_df = pitches_df.dropna(subset=cluster_features)

    scaler = StandardScaler()
    scaled_columns = [f"{feature}_Scaled" for feature in numerical_features]
    pitches_df[scaled_columns] = scaler.fit_transform(pitches_df[numerical_features])

    joblib.dump(scaler, "scaler.pkl")

    return pitches_df


def prepare_data(pitches_df, ids_df):
    pitches_df = clean_data(pitches_df, ids_df)
    pitches_df = encode_vars(pitches_df)
    pitches_df = add_fastball_specs(pitches_df)
    pitches_df = bucketize_plate_locations(pitches_df)
    pitches_df = scale_data(pitches_df)

    return pitches_df


def add_prev_pitch(pitches_df):
    pitches_df = pitches_df.sort_values(by=['game_date', 'pitcher', 'batter', 'PA_ID', 'pitch_number'])

    pitches_df['prev_pitch'] = (
        (pitches_df['pitch_number'] == pitches_df['pitch_number'].shift(-1) - 1) &
        (pitches_df['game_date'] == pitches_df['game_date'].shift(-1)) &
        (pitches_df['pitcher'] == pitches_df['pitcher'].shift(-1)) &
        (pitches_df['batter'] == pitches_df['batter'].shift(-1)) &
        (pitches_df['PA_ID'] == pitches_df['PA_ID'].shift(-1))
    )

    pitches_df['prev_pitch'] = pitches_df['prev_pitch'].shift(1)

    pitches_df['prev_pitch_RelSpeed'] = pitches_df['release_speed'].shift(1)
    pitches_df['prev_pitch_HorzBreak'] = pitches_df['pfx_x'].shift(1)
    pitches_df['prev_pitch_InducedVertBreak'] = pitches_df['pfx_z'].shift(1)
    pitches_df['prev_pitch_PlateLocSideBucket'] = pitches_df['PlateLocSideBucket'].shift(1)
    pitches_df['prev_pitch_PlateLocHeightBucket'] = pitches_df['PlateLocHeightBucket'].shift(1)
    pitches_df['prev_pitch_PitchCall'] = pitches_df['description'].shift(1)
    pitches_df['prev_pitch_PitchType'] = pitches_df['pitch_type'].shift(1)

    pitches_df['prev_pitch_PitchCall'] = pitches_df['prev_pitch_PitchCall'].apply(
        lambda x: 0 if x in ['ball', 'blocked_ball'] else
        1 if x == 'called_strike' else
        2 if x in ['foul_tip', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'] else
        3 if x in ['foul', 'foul_bunt'] else
        4
    )

    pitches_df['prev_pitch_SamePitch'] = (pitches_df['pitch_type'] == pitches_df['prev_pitch_PitchType']).astype(int)

    prev_pitch_invalid = pitches_df['prev_pitch'].isna() | (pitches_df['prev_pitch'] == False)

    columns_to_update = [
        'prev_pitch_RelSpeed', 'prev_pitch_HorzBreak', 'prev_pitch_InducedVertBreak',
        'prev_pitch_PlateLocSideBucket', 'prev_pitch_PlateLocHeightBucket', 'prev_pitch_PitchCall', 'prev_pitch_SamePitch'
    ]
    pitches_df[columns_to_update] = pitches_df[columns_to_update].where(~prev_pitch_invalid, np.nan)

    return pitches_df


def fit_gmms(pitches_df):
    gmm_models = {}

    for pitcher_side in ['L', 'R']:
        for pitch_group in [0, 1, 2]:
            group_df = pitches_df[
                (pitches_df['p_throws'] == pitcher_side) & 
                (pitches_df['PitchGroupEncoded'] == pitch_group)
            ]

            if group_df.empty:
                print(f"No data for PitcherThrows={pitcher_side} and PitchGroupEncoded={pitch_group}")
                continue

            gmm = GaussianMixture(n_components=num_clusters, random_state=100)
            gmm.fit(group_df[[f"{feature}_Scaled" for feature in numerical_features]])

            gmm_key = f"{pitcher_side}_PitchGroup_{pitch_group}"
            gmm_models[gmm_key] = gmm

    return gmm_models


def save_gmm_models(gmm_models, save_dir="gmm_models"):
    os.makedirs(save_dir, exist_ok=True)

    for model_key, gmm in gmm_models.items():
        model_filename = os.path.join(save_dir, f"{model_key}_gmm.pkl")
        
        joblib.dump(gmm, model_filename)
        print(f"Model {model_key} saved to {model_filename}")


def load_gmm_models():
    gmm_models = {}
    for model_key in model_keys:
        model_path = f"{model_dir}/{model_key}_gmm.pkl"
        gmm_models[model_key] = joblib.load(model_path)
    return gmm_models


def add_probabilities(pitches_df):
    gmm_models = load_gmm_models()

    prob_dfs = []

    pitches_df = pitches_df[pitches_df['p_throws'].isin(['L', 'R'])]
    grouped = pitches_df.groupby(['p_throws', 'PitchGroupEncoded'])

    for (pitcher_side, pitch_group), group in grouped:
        gmm_key = f"{pitcher_side}_PitchGroup_{int(pitch_group)}"

        if gmm_key not in gmm_models:
            continue

        gmm = gmm_models[gmm_key]

        group_indices = group.index

        scaled_columns = [f"{feature}_Scaled" for feature in numerical_features]
        scaled_data = pitches_df.loc[group_indices, scaled_columns]

        probabilities = gmm.predict_proba(scaled_data)

        if len(group_indices) != probabilities.shape[0]:
            raise ValueError(
                f"Mismatch between group indices ({len(group_indices)}) "
                f"and probabilities ({probabilities.shape[0]}). Check the input data."
            )

        temp_prob_df = pd.DataFrame(
            probabilities,
            columns=[f'prob_{i}' for i in range(probabilities.shape[1])],
            index=group_indices
        )

        prob_dfs.append(temp_prob_df)

    if prob_dfs:
        prob_df = pd.concat(prob_dfs)
        return pd.concat([pitches_df, prob_df], axis=1)
    else:
        raise ValueError("No probabilities were calculated. Check your GMM models or input data.")


def calculate_global_means(pitches_df):
    pitches_df = add_probabilities(pitches_df)
    results = []

    for (model, pitch_group), group in pitches_df.groupby(['p_throws', 'PitchGroupEncoded']):
        for cluster in range(num_clusters):
            cluster_col = f'prob_{cluster}'

            if cluster_col not in group.columns:
                continue
            
            group = group.drop_duplicates(subset=['p_throws', 'PitchGroupEncoded', cluster_col])

            cluster_weights = group[cluster_col]
            if cluster_weights.sum() == 0:
                continue

            weighted_mean = (
                (group['delta_run_exp'] * cluster_weights).sum() /
                cluster_weights.sum()
            )

            results.append({
                'Model': model,
                'PitchGroup': pitch_group,
                'Cluster': cluster,
                'GlobalMean': weighted_mean
            })

    global_means_df = pd.DataFrame(results)
    global_means_df.to_csv('global_means.csv', index=False)

    return global_means_df


def calculate_shrunken_means(pitches_df, global_means, batter=None):
    results = []

    if pitches_df['game_date'].dtype == 'object':
        pitches_df['game_date'] = pd.to_datetime(pitches_df['game_date'])
    current_day = pitches_df['game_date'].max().normalize()

    pitches_df['DayDiff'] = (current_day - pitches_df['game_date'].dt.normalize()).dt.days
    pitches_df['DayWeight'] = 0.999 ** pitches_df['DayDiff']
    pitches_df['Model'] = pitches_df['p_throws']
    global_means['Model'] = global_means['Model'].astype(str)

    for (batter_id, pitcher_throws), group in pitches_df.groupby(['batter', 'p_throws']):
        if batter is not None and batter_id != batter:
            continue

        cluster_stats = []

        for cluster in range(num_clusters):
            prob_col = f'prob_{cluster}'

            if prob_col not in group.columns:
                raise KeyError(f"Missing column: {prob_col}")

            time_weighted_probs = group[prob_col].fillna(0) * group['DayWeight']
            total_weights = time_weighted_probs.sum()

            if total_weights > 0:
                weighted_mean = (group['delta_run_exp'] * time_weighted_probs).sum() / total_weights
            else:
                weighted_mean = 0

            global_mean = global_means.loc[
                (global_means['Cluster'] == cluster) & (global_means['Model'] == pitcher_throws),
                'GlobalMean'
            ].values[0]

            shrunken_mean = (
                (total_weights * weighted_mean + pseudo_sample_size * global_mean) /
                (total_weights + pseudo_sample_size)
            )

            cluster_stats.append({
                'Cluster': cluster,
                'ShrunkenMean_DeltaRunValue': shrunken_mean
            })

        cluster_stats_df = pd.DataFrame(cluster_stats)
        cluster_stats_df['batter'] = batter_id
        cluster_stats_df['Model'] = pitcher_throws
        results.append(cluster_stats_df)

    combined_stats = pd.concat(results).reset_index(drop=True)
    
    pivoted_values = combined_stats.pivot(
        index=['batter', 'Model'],
        columns='Cluster',
        values='ShrunkenMean_DeltaRunValue'
    ).reset_index()

    pivoted_values.columns = [
        f'DeltaRunValue_{col}' if isinstance(col, int) else col for col in pivoted_values.columns
    ]

    return pitches_df, pivoted_values


def compute_batter_stuff_value(pitches_df, pivoted_values):
    merged_df = pitches_df.merge(
        pivoted_values,
        how='left',
        left_on=['batter', 'Model'],
        right_on=['batter', 'Model']
    )
    
    weighted_sum = 0
    for i in range(num_clusters):
        prob_col = f'prob_{i}' 
        value_col = f'DeltaRunValue_{i}' 
        merged_df[prob_col] = merged_df[prob_col].fillna(0)
        merged_df[value_col] = merged_df[value_col].fillna(0)
        weighted_sum += merged_df[prob_col] * merged_df[value_col]

    merged_df['BatterStuffValue'] = weighted_sum

    return merged_df

def calculate_VRA(vy0, ay, release_extension, vz0, az):
    vy_s = -np.sqrt(vy0**2 - 2 * ay * (60.5 - release_extension - 50))
    t_s = (vy_s - vy0) / ay
    vz_s = vz0 - az * t_s
    VRA = -np.arctan(vz_s / vy_s) * (180 / np.pi)
    return VRA

def calculate_HRA(vy0, ay, release_extension, vx0, ax):
    vy_s = -np.sqrt(vy0**2 - 2 * ay * (60.5 - release_extension - 50))
    t_s = (vy_s - vy0) / ay
    vx_s = vx0 - ax * t_s
    HRA = -np.arctan(vx_s / vy_s) * (180 / np.pi)
    return HRA