import numpy as np

columns_to_keep = [
    'PitchNo', 'Date', 'Pitcher', 'PitcherId', 'PitcherThrows', 'PitcherTeam',
    'TaggedPitchType', 'RelSpeed', 'VertRelAngle', 'HorzRelAngle', 'SpinRate',
    'SpinAxis', 'RelHeight', 'RelSide', 'Extension', 'InducedVertBreak', 'HorzBreak',
    'PlateLocHeight', 'PlateLocSide', 'ax0', 'ay0', 'az0', 'PitchUID', 'BatterId',
    'Batter', 'HitType', 'ExitSpeed', 'BatterSide', 'Angle', 'Direction', 'PAofInning',
    'BatterTeam', 'Inning', 'Top/Bottom', 'Outs', 'Balls', 'Strikes', 'AutoPitchType', 
    'PitchCall', 'PlayResult', 'OutsOnPlay', 'RunsScored', 'HomeTeam', 'AwayTeam', 'Level',
    'League', 'GameUID', 'UTCDateTime'
]

spray_bins = [-np.inf, -67.5, -52.5, -37.5, -22.5, -7.5, 7.5, 22.5, 37.5, 52.5, 67.5, np.inf]
spray_labels = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75]

bb_features = ['ExitSpeed', 'Angle', 'DirectionBucket']
bb_target = 'DeltaRunValue'

batter_league_encoding = {
    'Team': 0,
    'SEC': 0,
    'ACC': 1,
    'SBELT': 2,
    'BIG12': 3,
    'BIG10': 4,
    'PAC12': 5,
    'MVC': 6,
    'WAC': 7,
    'CUSA': 8,
    'AMER': 9,
    'CAA': 10,
    'SOCON': 11,
    'BSOU': 12,
    'WCC': 13,
    'A10': 14,
    'ASUN': 15,
    'BW': 16,
    'MW': 17,
    'BEAST': 18,
    'SLAND': 19,
    'PAT': 20,
    'IVY': 21,
    'OVC': 22,
    'SUM': 23,
    'HORZ': 24,
    'NEC': 25,
    'AMEAST': 26,
    'MAAC': 27,
    'MAC': 28,
}

platoon_state_mapping = {
    ('Left', 'Left'): 0,
    ('Left', 'Right'): 1,
    ('Right', 'Left'): 2,
    ('Right', 'Right'): 3,
}

count_conditions = [
    [(0, 0), (1, 0), (1, 1), (2, 1), (3, 2)],
    [(0, 2), (0, 1), (1, 2), (2, 2)],
    [(2, 0), (3, 0), (3, 1)]
]

count_values = [0, 1, 2]

pitch_group_mapping = {
    'Fastball': 0, 'Sinker': 0, 'FourSeamFastBall': 0, 'Four-Seam': 0, 'TwoSeamFastBall': 0,
    'Slider': 1, 'Curveball': 1, 'Cutter': 1,
    'ChangeUp': 2, 'Changeup': 2, 'Splitter': 2, 'Knuckleball': 2
}

side_buckets = np.array([-1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8])
height_buckets = np.array([0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2])

num_clusters = 25

numerical_features = ['RelSpeed', 'ax0', 'az0', 'PlateLocSide', 'PlateLocHeight']
cluster_features = ['RelSpeed', 'ax0', 'az0', 'PitcherThrows', 'PitchGroupEncoded', 'PlateLocSide', 'PlateLocHeight']

model_dir = 'gmm_models'
model_keys = [
    'Left_PitchGroup_0', 'Left_PitchGroup_1', 'Left_PitchGroup_2', 
    'Right_PitchGroup_0', 'Right_PitchGroup_1', 'Right_PitchGroup_2'
]

pseudo_sample_size = 50

rv_features = ['RelSpeed', 'SpinRate', 'SpinAxis', 'RelHeight', 'RelSide', 'Extension', 
            'PlateLocSideBucket', 'PlateLocHeightBucket', 'ax0', 'ay0', 'az0', 
            'avg_fb_RelSpeed', 'avg_fb_ax0', 'avg_fb_az0', 'avg_fb_RelHeight', 'avg_fb_RelSide',
            'Year', 'BatterLeagueEncoded', 'PlatoonStateEncoded', 'CountEncoded', 
            'PitchGroupEncoded', 'BatterStuffValue']
rv_target = 'DeltaRunValue'

median_features = [
    'RelSpeed', 'SpinRate', 'SpinAxis', 'RelHeight', 'RelSide', 'Extension', 
    'ax0', 'ay0', 'az0', 'avg_fb_RelSpeed', 'avg_fb_ax0', 'avg_fb_az0', 'avg_fb_RelHeight', 'avg_fb_RelSide'
]