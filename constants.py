import numpy as np

columns_to_keep = [
    'pitch_number', 'game_date', 'player_name', 'pitcher', 'p_throws',
    'pitch_type', 'release_speed', 'release_spin_rate',
    'spin_axis', 'release_pos_z', 'release_pos_x', 'release_extension', 'pfx_z', 'pfx_x',
    'plate_z', 'plate_x', 'ax', 'ay', 'az', 'batter', 'stand',
    'inning', 'inning_topbot', 'at_bat_number', 'balls', 'strikes',
    'description', 'events', 'home_team', 'game_pk', 'game_type', 'delta_run_exp', 'VRA', 'HRA'
]

spray_bins = [-np.inf, -67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5, np.inf]
spray_labels = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]

bb_features = ['ExitSpeed', 'Angle', 'DirectionBucket']
bb_target = 'delta_run_exp'

platoon_state_mapping = {
    ('L', 'L'): 0,
    ('L', 'R'): 1,
    ('R', 'L'): 2,
    ('R', 'R'): 3,
}

count_conditions = [
    [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)],
    [(0, 2), (0, 1), (1, 2), (2, 2)],
    [(2, 0), (3, 0), (3, 1)]
]

count_values = [0, 1, 2]

pitch_group_mapping = {
    'FF': 0, 'SI': 0, 'FourSeamFastBall': 0, 'Four-Seam': 0, 'TwoSeamFastBall': 0,
    'SL': 1, 'CU': 1, 'FC': 1, 'ST': 1, 'KC': 1, 'CS': 1, 'SV': 1,
    'CH': 2, 'FS': 2, 'KN': 2, 'EP': 2, 'FO': 2, 'SC': 2
}

side_buckets = np.array([-1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8])
height_buckets = np.array([0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2])

num_clusters = 25

numerical_features = ['release_speed', 'ax', 'az', 'plate_x', 'plate_z']
cluster_features = ['release_speed', 'ax', 'az', 'p_throws', 'PitchGroupEncoded', 'plate_x', 'plate_z']

model_dir = 'gmm_models'
model_keys = [
    'L_PitchGroup_0', 'L_PitchGroup_1', 'L_PitchGroup_2', 
    'R_PitchGroup_0', 'R_PitchGroup_1', 'R_PitchGroup_2'
]

pseudo_sample_size = 50

rv_features = ['release_speed', 'release_spin_rate', 'spin_axis', 'release_pos_z', 'release_pos_x', 'release_extension', 
            'PlateLocSideBucket', 'PlateLocHeightBucket', 'ax', 'ay', 'az', 
            'avg_fb_RelSpeed', 'avg_fb_ax0', 'avg_fb_az0', 'avg_fb_RelHeight', 'avg_fb_RelSide',
            'year', 'PlatoonStateEncoded', 'CountEncoded', 'PitchGroupEncoded', 'BatterStuffValue']
rv_target = 'delta_run_exp'

median_features = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'release_pos_z', 'release_pos_x', 'release_extension', 
    'ax', 'ay', 'az', 'avg_fb_RelSpeed', 'avg_fb_ax0', 'avg_fb_az0', 'avg_fb_RelHeight', 'avg_fb_RelSide'
]

pitch_type_name = {
    "FF": "Four-Seam Fastball",
    "SL": "Slider",
    "SI": "Sinker",
    "CH": "Changeup",
    "FC": "Cutter",
    "CU": "Curveball",
    "ST": "Sweeper",
    "FS": "Splitter",
    "KC": "Knuckle Curve",
    "SV": "Slurve",
    "EP": "Eephus",
    "KN": "Knuckleball",
    "FO": "Forkball",
    "SC": "Screwball",
    "CS": "Slow Curve"
}
