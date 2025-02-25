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
    bb_features,
    spray_bins,
    spray_labels,
    batter_league_encoding,
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

def clean_data(pitches_df, game_only=False):
    pitches_df = pitches_df[columns_to_keep]
    pitches_df = pitches_df.dropna(subset=['RelSpeed', 'ax0', 'az0', 'PlateLocSide', 'PlateLocHeight'])

    batter_side_mode = pitches_df.groupby(['BatterId', 'PitcherThrows'])['BatterSide'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
    pitches_df['BatterSide'] = pitches_df.apply(lambda row: batter_side_mode.get((row['BatterId'], row['PitcherThrows']), row['BatterSide']) if pd.isna(row['BatterSide']) else row['BatterSide'], axis=1)

    pitcher_throws_mode = pitches_df.groupby(['PitcherId', 'BatterSide'])['PitcherThrows'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
    pitches_df['PitcherThrows'] = pitches_df.apply(lambda row: pitcher_throws_mode.get((row['PitcherId'], row['BatterSide']), row['PitcherThrows']) if pd.isna(row['PitcherThrows']) else row['PitcherThrows'], axis=1)

    pitches_df['Date'] = pd.to_datetime(pitches_df['Date'], errors='coerce')
    pitches_df = pitches_df.dropna(subset=['Date'])

    if game_only:
        date_ranges = [
            ('2025-02-14', '2025-07-01'),
            ('2024-02-16', '2024-06-24'),
            ('2023-02-17', '2023-06-26'),
            ('2022-02-18', '2022-06-22')
        ]
        
        pitches_df = pd.concat([
            pitches_df[(pitches_df['Date'] >= pd.to_datetime(start)) & (pitches_df['Date'] <= pd.to_datetime(end))]
            for start, end in date_ranges
        ])
        
        pitches_df = pitches_df[(pitches_df['PitcherTeam'] != 'GEO_PRA') & (pitches_df['BatterTeam'] != 'GEO_PRA')]

    pitches_df = pitches_df[(pitches_df['Level'] == 'D1') | (pitches_df['Level'] == 'TeamExclusive') | (pitches_df['Level'] == 'D3')]

    pitches_df = pitches_df[(pitches_df['Balls'] <= 3) & (pitches_df['Strikes'] <= 2)]
    pitches_df = pitches_df[pitches_df['PitcherThrows'].isin(['Left', 'Right'])]

    pitches_df.loc[pitches_df['PlayResult'] == 'FieldersChoice', 'PlayResult'] = 'Out'

    pitches_df['PA_ID'] = (
        pitches_df.groupby(['GameUID', 'Inning', 'Top/Bottom', 'PAofInning']).ngroup()
    )
    
    pitches_df.loc[
        (pitches_df['PitchCall'].isin(['BallCalled', 'BallinDirt', 'BallIntentional'])) &
        (pitches_df['Balls'] == 3),
        'PlayResult'
    ] = 'Walk'

    pitches_df.loc[
        (pitches_df['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging'])) &
        (pitches_df['Strikes'] == 2),
        'PlayResult'
    ] = 'StrikeOut'

    pitches_df.loc[
        pitches_df['PitchCall'] == 'HitByPitch',
        'PlayResult'
    ] = 'HitByPitch'

    return pitches_df


def calculate_game_metrics(pitches_df):
    date_ranges = [
        ('2025-02-14', '2025-07-01'),
        ('2024-02-16', '2024-06-24'),
        ('2023-02-17', '2023-06-26'),
        ('2022-02-18', '2022-06-22')
    ]
    
    in_game_df = pd.concat([
        pitches_df[(pitches_df['Date'] >= pd.to_datetime(start)) & (pitches_df['Date'] <= pd.to_datetime(end))]
        for start, end in date_ranges
    ])
    
    in_game_df = in_game_df[(in_game_df['PitcherTeam'] != 'GEO_PRA') & (in_game_df['BatterTeam'] != 'GEO_PRA')]

    grouped = in_game_df.groupby(['GameUID', 'Top/Bottom'])

    game_df = grouped.agg(
        sum_runs=('RunsScored', 'sum'),
        num_single=('PlayResult', lambda x: (x == 'Single').sum()),
        num_double=('PlayResult', lambda x: (x == 'Double').sum()),
        num_triple=('PlayResult', lambda x: (x == 'Triple').sum()),
        num_homerun=('PlayResult', lambda x: (x == 'HomeRun').sum()),
        num_walk=('PlayResult', lambda x: (x == 'Walk').sum()),
        num_hitbypitch=('PlayResult', lambda x: (x == 'HitByPitch').sum()),
    ).reset_index()

    return game_df


def predict_pa_run_value(game_df):    
    game_df['sum_runs'] = pd.to_numeric(game_df['sum_runs'], errors='coerce')
    game_df = game_df.dropna(subset=['sum_runs'])
    
    X = game_df[['num_single', 'num_double', 'num_triple', 'num_homerun', 'num_walk', 'num_hitbypitch']]
    y = game_df['sum_runs']

    model = LinearRegression()
    model.fit(X, y)

    coefficients = pd.Series(model.coef_, index=X.columns)

    return coefficients


def calculate_out_pa_run_value(pitches_df, coefficients):
    total_outcomes = pitches_df['PlayResult'].isin(
        ['Out', 'StrikeOut', 'Single', 'Double', 'Triple', 'HomeRun', 'Walk', 'HitByPitch']
    ).sum()

    outcome_probs = {
        play_result: (pitches_df['PlayResult'] == play_result).sum() / total_outcomes
        for play_result in ['Single', 'Double', 'Triple', 'HomeRun', 'Walk', 'HitByPitch']
    }

    pa_run_value_out = sum(
        outcome_probs[play_result] * -coefficients[f'num_{play_result.lower()}']
        for play_result in outcome_probs
    )

    return pa_run_value_out


def assign_pa_run_values(pitches_df, coefficients, pa_run_value_out):
    play_result_to_coef = {
        'Single': coefficients['num_single'],
        'Double': coefficients['num_double'],
        'Triple': coefficients['num_triple'],
        'HomeRun': coefficients['num_homerun'],
        'Walk': coefficients['num_walk'],
        'HitByPitch': coefficients['num_hitbypitch'],
    }

    pitches_df['PARunValue'] = pitches_df['PlayResult'].map(play_result_to_coef)
    pitches_df.loc[pitches_df['PlayResult'].isin(['Out', 'StrikeOut']), 'PARunValue'] = pa_run_value_out
    pitches_df.loc[~pitches_df['PlayResult'].isin(play_result_to_coef.keys() | {'Out', 'StrikeOut'}), 'PARunValue'] = np.nan

    pitches_df['PARunValue'] = pitches_df.groupby('PA_ID')['PARunValue'].transform('max')
    pitches_df['PrePitchRunValue'] = pitches_df.groupby(['Balls', 'Strikes'])['PARunValue'].transform('mean')

    return pitches_df


def assign_post_pitch_run_value(pitches_df):
    grouped_means = pitches_df.groupby(['Balls', 'Strikes'])['PARunValue'].mean()

    pitches_df['PostPitchRunValue'] = np.nan

    mask_ball = pitches_df['PitchCall'].isin(['BallCalled', 'BallinDirt'])
    pitches_df.loc[mask_ball, 'PostPitchRunValue'] = pitches_df[mask_ball].apply(
        lambda row: grouped_means.get((row['Balls'] + 1, row['Strikes']), np.nan)
        if row['Balls'] + 1 <= 3 else np.nan, axis=1
    )

    mask_strike = pitches_df['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging'])
    pitches_df.loc[mask_strike, 'PostPitchRunValue'] = pitches_df[mask_strike].apply(
        lambda row: grouped_means.get((row['Balls'], row['Strikes'] + 1), np.nan)
        if row['Strikes'] + 1 <= 2 else np.nan, axis=1
    )

    mask_foul = pitches_df['PitchCall'].isin(['FoulBall', 'FoulBallNotFieldable', 'FoulBallFieldable'])
    pitches_df.loc[mask_foul, 'PostPitchRunValue'] = pitches_df[mask_foul].apply(
        lambda row: grouped_means.get((row['Balls'], row['Strikes'] + 1), np.nan)
        if row['Strikes'] + 1 <= 2 else grouped_means.get((row['Balls'], row['Strikes']), np.nan), axis=1
    )

    pitches_df.loc[pitches_df['PostPitchRunValue'].isna(), 'PostPitchRunValue'] = pitches_df['PARunValue']
    pitches_df['DeltaRunValue'] = pitches_df['PostPitchRunValue'] - pitches_df['PrePitchRunValue']

    return pitches_df


def add_run_value(pitches_df):
    game_df = calculate_game_metrics(pitches_df)

    coefficients = predict_pa_run_value(game_df)
    pa_run_value_out = calculate_out_pa_run_value(pitches_df, coefficients)

    pitches_df = assign_pa_run_values(pitches_df, coefficients, pa_run_value_out)
    pitches_df = assign_post_pitch_run_value(pitches_df)

    return pitches_df


def bucketize_spray_angle(pitches_df):
    pitches_df['DirectionBucket'] = pd.cut(
        pitches_df['Direction'], bins=spray_bins, labels=spray_labels
    ).astype(pd.Int32Dtype())

    return pitches_df


def apply_batted_ball_model(pitches_df):
    batted_ball_model = joblib.load('batted_ball_model.pkl')
    pitches_df = bucketize_spray_angle(pitches_df)

    batted_ball_df = pitches_df[pitches_df['PitchCall'] == 'InPlay']
    batted_ball_df = batted_ball_df.dropna(subset=['ExitSpeed', 'Angle', 'DirectionBucket'])

    batted_ball_df['PredictedDeltaRunValue'] = batted_ball_model.predict(batted_ball_df[bb_features])
    pitches_df.loc[batted_ball_df.index, 'DeltaRunValue'] = batted_ball_df['PredictedDeltaRunValue']

    return pitches_df


def encode_vars(pitches_df):
    pitches_df['Date'] = pd.to_datetime(pitches_df['Date'])
    pitches_df['Year'] = pitches_df['Date'].dt.year

    home_team_year_mode_league = (
        pitches_df.groupby(['HomeTeam', 'Year'])['League']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        .reset_index()
        .rename(columns={'League': 'BatterLeague'})
    )

    pitches_df = pitches_df.merge(
        home_team_year_mode_league,
        left_on=['BatterTeam', 'Year'],
        right_on=['HomeTeam', 'Year'],
        how='left'
    )

    pitches_df['BatterLeagueEncoded'] = pitches_df['BatterLeague'].map(batter_league_encoding)

    pitches_df['PlatoonStateEncoded'] = pitches_df[['PitcherThrows', 'BatterSide']].apply(
        lambda x: platoon_state_mapping.get((x['PitcherThrows'], x['BatterSide'])), axis=1
    )

    pitches_df['CountEncoded'] = np.select(
        [pitches_df[['Balls', 'Strikes']].apply(tuple, axis=1).isin(condition) for condition in count_conditions],
        count_values,
        default=np.nan
    )

    pitches_df['TaggedPitchType'] = pitches_df['TaggedPitchType'].replace({
        'FourSeamFastBall': 'Fastball',
        'TwoSeamFastBall': 'Sinker'
    })

    pitches_df['PitchType'] = np.where(
        pitches_df['TaggedPitchType'] != 'Undefined',
        pitches_df['TaggedPitchType'],
        pitches_df['AutoPitchType']
    )

    pitches_df['PitchGroupEncoded'] = pitches_df['PitchType'].map(pitch_group_mapping)

    return pitches_df


def add_fastball_specs(pitches_df):
    pitches_df['UTCDateTime'] = pd.to_datetime(pitches_df['UTCDateTime'])
    
    pitches_df['Season'] = pitches_df['UTCDateTime'].dt.year

    common_pitch_type = (
        pitches_df[pitches_df['PitchGroupEncoded'] == 0]
        .groupby(['PitcherId', 'BatterSide', 'Season'])['TaggedPitchType']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={'TaggedPitchType': 'MostCommonPitchType'})
    )

    pitches_df = pitches_df.merge(common_pitch_type, on=['PitcherId', 'BatterSide', 'Season'], how='left')

    subset_df = pitches_df[pitches_df['TaggedPitchType'] == pitches_df['MostCommonPitchType']]

    medians = (
        subset_df
        .groupby(['PitcherId', 'BatterSide', 'Season'])
        [['RelSpeed', 'ax0', 'az0', 'RelHeight', 'RelSide']]
        .median()
        .reset_index()
        .rename(columns={
            'RelSpeed': 'avg_fb_RelSpeed',
            'ax0': 'avg_fb_ax0',
            'az0': 'avg_fb_az0',
            'RelHeight': 'avg_fb_RelHeight',
            'RelSide': 'avg_fb_RelSide'
        })
    )

    pitches_df = pitches_df.merge(medians, on=['PitcherId', 'BatterSide', 'Season'], how='left')

    return pitches_df


def bucketize_plate_locations(pitches_df):
    pitches_df['PlateLocSideBucket'] = pitches_df['PlateLocSide'].apply(lambda x: side_buckets[np.argmin(np.abs(side_buckets - x))])
    pitches_df['PlateLocHeightBucket'] = pitches_df['PlateLocHeight'].apply(lambda x: height_buckets[np.argmin(np.abs(height_buckets - x))])

    return pitches_df


def scale_data(pitches_df, save=False):
    pitches_df['BatterLefty'] = (pitches_df['BatterSide'] == 'Left').astype(int)
    pitches_df['PitcherLefty'] = (pitches_df['PitcherThrows'] == 'Left').astype(int)

    pitches_df = pitches_df.dropna(subset=cluster_features)

    scaler = StandardScaler()
    scaled_columns = [f"{feature}_Scaled" for feature in numerical_features]
    pitches_df[scaled_columns] = scaler.fit_transform(pitches_df[numerical_features])

    if save:
        joblib.dump(scaler, "scaler.pkl")

    return pitches_df


def prepare_data(pitches_df, game_only=False):
    pitches_df = clean_data(pitches_df, game_only)
    pitches_df = add_run_value(pitches_df)

    pitches_df = apply_batted_ball_model(pitches_df)
    pitches_df = encode_vars(pitches_df)

    pitches_df = add_fastball_specs(pitches_df)
    pitches_df = bucketize_plate_locations(pitches_df)

    pitches_df = scale_data(pitches_df)

    return pitches_df


def add_prev_pitch(pitches_df):
    pitches_df = pitches_df.sort_values(by=['Date', 'PitcherId', 'BatterId', 'PA_ID', 'PitchNo'])

    pitches_df['prev_pitch'] = (
        (pitches_df['PitchNo'] == pitches_df['PitchNo'].shift(-1) - 1) &
        (pitches_df['Date'] == pitches_df['Date'].shift(-1)) &
        (pitches_df['PitcherId'] == pitches_df['PitcherId'].shift(-1)) &
        (pitches_df['BatterId'] == pitches_df['BatterId'].shift(-1)) &
        (pitches_df['PA_ID'] == pitches_df['PA_ID'].shift(-1))
    )

    pitches_df['prev_pitch'] = pitches_df['prev_pitch'].shift(1)

    pitches_df['prev_pitch_RelSpeed'] = pitches_df['RelSpeed'].shift(1)
    pitches_df['prev_pitch_HorzBreak'] = pitches_df['HorzBreak'].shift(1)
    pitches_df['prev_pitch_InducedVertBreak'] = pitches_df['InducedVertBreak'].shift(1)
    pitches_df['prev_pitch_PlateLocSideBucket'] = pitches_df['PlateLocSideBucket'].shift(1)
    pitches_df['prev_pitch_PlateLocHeightBucket'] = pitches_df['PlateLocHeightBucket'].shift(1)
    pitches_df['prev_pitch_PitchCall'] = pitches_df['PitchCall'].shift(1)
    pitches_df['prev_pitch_PitchType'] = pitches_df['PitchType'].shift(1)

    pitches_df['prev_pitch_PitchCall'] = pitches_df['prev_pitch_PitchCall'].apply(
        lambda x: 0 if x in ['BallCalled', 'BallinDirt'] else
        1 if x == 'StrikeCalled' else
        2 if x == 'StrikeSwinging' else
        3 if x in ['FoulBall', 'FoulBallNotFieldable', 'FoulBallFieldable'] else
        4
    )

    pitches_df['prev_pitch_SamePitch'] = (pitches_df['PitchType'] == pitches_df['prev_pitch_PitchType']).astype(int)

    prev_pitch_invalid = pitches_df['prev_pitch'].isna() | (pitches_df['prev_pitch'] == False)

    columns_to_update = [
        'prev_pitch_RelSpeed', 'prev_pitch_HorzBreak', 'prev_pitch_InducedVertBreak',
        'prev_pitch_PlateLocSideBucket', 'prev_pitch_PlateLocHeightBucket', 'prev_pitch_PitchCall', 'prev_pitch_SamePitch'
    ]
    pitches_df[columns_to_update] = pitches_df[columns_to_update].where(~prev_pitch_invalid, np.nan)

    return pitches_df


def fit_gmms(pitches_df):
    gmm_models = {}

    for pitcher_side in ['Left', 'Right']:
        for pitch_group in [0, 1, 2]:
            group_df = pitches_df[
                (pitches_df['PitcherThrows'] == pitcher_side) & 
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

    pitches_df = pitches_df[pitches_df['PitcherThrows'].isin(['Left', 'Right'])]
    grouped = pitches_df.groupby(['PitcherThrows', 'PitchGroupEncoded'])

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

    for (model, pitch_group), group in pitches_df.groupby(['PitcherThrows', 'PitchGroupEncoded']):
        for cluster in range(num_clusters):
            cluster_col = f'prob_{cluster}'

            if cluster_col not in group.columns:
                continue
            
            group = group.drop_duplicates(subset=['PitcherThrows', 'PitchGroupEncoded', cluster_col])

            cluster_weights = group[cluster_col]
            if cluster_weights.sum() == 0:
                continue

            weighted_mean = (
                (group['DeltaRunValue'] * cluster_weights).sum() /
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

    if pitches_df['UTCDateTime'].dtype == 'object':
        pitches_df['UTCDateTime'] = pd.to_datetime(pitches_df['UTCDateTime'])
    current_day = pitches_df['UTCDateTime'].max().normalize()

    pitches_df['DayDiff'] = (current_day - pitches_df['UTCDateTime'].dt.normalize()).dt.days
    pitches_df['DayWeight'] = 0.999 ** pitches_df['DayDiff']
    pitches_df['Model'] = pitches_df['PitcherThrows']
    global_means['Model'] = global_means['Model'].astype(str)

    prob_columns = [f'prob_{i}' for i in range(num_clusters)]

    for (batter_id, pitcher_throws), group in pitches_df.groupby(['BatterId', 'PitcherThrows']):
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
                weighted_mean = (group['DeltaRunValue'] * time_weighted_probs).sum() / total_weights
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
        cluster_stats_df['BatterId'] = batter_id
        cluster_stats_df['Model'] = pitcher_throws
        results.append(cluster_stats_df)

    combined_stats = pd.concat(results).reset_index(drop=True)
    
    pivoted_values = combined_stats.pivot(
        index=['BatterId', 'Model'],
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
        left_on=['BatterId', 'Model'],
        right_on=['BatterId', 'Model']
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
