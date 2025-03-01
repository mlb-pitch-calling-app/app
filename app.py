import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from itertools import product
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import joblib
from sklearn.preprocessing import StandardScaler
import helper_functions as hf
import warnings
import time
from constants import(
    platoon_state_mapping,
    count_values,
    median_features,
    cluster_features,
    numerical_features
)
warnings.filterwarnings('ignore')

@st.cache_data(ttl=900)
def load_data():
    pitches_df = pd.read_csv('mlb_pitches.csv')
    global_means = pd.read_csv("global_means.csv")

    pitches_df = hf.prepare_data(pitches_df)

    return pitches_df, global_means

@st.cache_resource(ttl=900)
def load_models():
    rv_model = joblib.load("rv_model.pkl")
    prev_pitch_model = joblib.load("prev_pitch_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return rv_model, prev_pitch_model, scaler

pitches_df, global_means = load_data()
rv_model, prev_pitch_model, scaler = load_models()

star_ratings = [
    (-0.045, "★★★★★"), (-0.035, "★★★★☆"), (-0.025, "★★★☆☆"),
    (-0.015, "★★☆☆☆"), (-0.005, "★☆☆☆☆"), (0.005, "☆☆☆☆☆"),
    (0.015, "★☆☆☆☆"), (0.025, "★★☆☆☆"), (0.035, "★★★☆☆"),
    (0.045, "★★★★☆"), (float("inf"), "★★★★★")
]

cmap = mcolors.LinearSegmentedColormap.from_list("green_to_white_to_red", 
                                                [(0, "green"), (0.2, "white"), (0.8, "white"), (1, "red")], 
                                                N=256)

def simulate_synthetic_dataframe(recent_rows, batter_df, pitch_type, balls, strikes):
    synthetic_data = pd.DataFrame(
        list(product(side_buckets, height_buckets, count_values)),
        columns=['PlateLocSideBucket', 'PlateLocHeightBucket', 'CountEncoded']
    )

    synthetic_data['pitch_type'] = pitch_type

    pitcher_throws_mode = (recent_rows['p_throws'].mode().iloc[0])

    synthetic_data['p_throws'] = pitcher_throws_mode

    batter_side_mode = (
        batter_df.loc[batter_df['p_throws'] == pitcher_throws_mode]
        .sort_values(by=['game_date', 'game_pk'], ascending=False)
        .head(100)
        .BatterSide
        .mode().iloc[0]
    )

    platoon_state_encoded = platoon_state_mapping[(pitcher_throws_mode, batter_side_mode)]
    synthetic_data['PlatoonStateEncoded'] = platoon_state_encoded

    synthetic_data['BatterLeagueEncoded'] = (
        batter_df.sort_values(['game_date', 'game_pk'], ascending=False)
        .iloc[0]['BatterLeagueEncoded']
    )

    medians = recent_rows[recent_rows['pitch_type'] == pitch_type].sort_values(by=['game_date', 'game_pk'], ascending=False)[median_features].median().to_dict()

    for feature, median_value in medians.items():
        synthetic_data[feature] = median_value
    
    pitch_group_mode = recent_rows.loc[recent_rows['pitch_type'] == pitch_type, 'PitchGroupEncoded'].mode()[0]
    synthetic_data['PitchGroupEncoded'] = pitch_group_mode

    if not is_first_time and st.session_state['selected_zone'] is not None:
        synthetic_data['prev_pitch_RelSpeed'] = recent_rows['prev_pitch_RelSpeed']
        synthetic_data['prev_pitch_HorzBreak'] = recent_rows['prev_pitch_HorzBreak']
        synthetic_data['prev_pitch_InducedVertBreak'] = recent_rows['prev_pitch_InducedVertBreak']
        synthetic_data['prev_pitch_PlateLocSideBucket'] = recent_rows['prev_pitch_PlateLocSideBucket']
        synthetic_data['prev_pitch_PlateLocHeightBucket'] = recent_rows['prev_pitch_PlateLocHeightBucket']
        synthetic_data['prev_pitch_PitchCall'] = recent_rows['prev_pitch_PitchCall']
        synthetic_data['prev_pitch_SamePitch'] = recent_rows['prev_pitch_SamePitch']

    synthetic_data['balls'] = balls
    synthetic_data['strikes'] = strikes

    return synthetic_data


def calculate_batter_metrics(synthetic_df, batter_df):
    platoon_state_encoded = synthetic_df['PlatoonStateEncoded'].iloc[0]
    pitch_group_encoded = synthetic_df['PitchGroupEncoded'].iloc[0]
    pitcher_throws = synthetic_df['p_throws'].iloc[0]

    batter_id = batter_df['batter'].mode().iloc[0]

    synthetic_df['plate_x'] = (synthetic_df['PlateLocSideBucket'].astype(float))
    synthetic_df['plate_z'] = (synthetic_df['PlateLocHeightBucket'].astype(float))

    scaled_columns = [f"{feature}_Scaled" for feature in numerical_features]
    synthetic_df[scaled_columns] = scaler.transform(synthetic_df[numerical_features])

    synthetic_df = hf.add_probabilities(synthetic_df)
    batter_df = hf.add_probabilities(batter_df[batter_df['p_throws'] == pitcher_throws])

    _, pivoted_values = hf.calculate_shrunken_means(
        batter_df, global_means
    )

    synthetic_df['batter'] = batter_id
    synthetic_df['Model'] = pitcher_throws

    synthetic_df = hf.compute_batter_stuff_value(synthetic_df, pivoted_values)

    return synthetic_df


def generate_individual_figs(recent_rows, batter_df, model, balls, strikes):
    columns_to_drop = [col for col in recent_rows.columns if col.startswith('delta_run_exp_') or col.startswith('prob_')]
    recent_rows = recent_rows.drop(columns=columns_to_drop)

    columns_to_drop = [col for col in batter_df.columns if col.startswith('delta_run_exp_') or col.startswith('prob_')]
    batter_df = batter_df.drop(columns=columns_to_drop)

    if not recent_rows.index.is_unique:
        recent_rows = recent_rows.reset_index(drop=True)

    if not batter_df.index.is_unique:
        batter_df = batter_df.reset_index(drop=True)
    
    ### Issue: No pitcher_id or batter_id variables?
    if not recent_rows.empty:
        pitcher_throws_mode = recent_rows['p_throws'].mode().iloc[0]
    else:
        raise ValueError(f"No matching rows found for pitcher: {pitcher_id}")

    if not batter_df.empty:
        batter_side_mode = batter_df['stand'].mode().iloc[0]
    else:
        raise ValueError(f"No matching rows found for batter: {batter_id}")

    platoon_state_encoded = platoon_state_mapping[(pitcher_throws_mode, batter_side_mode)]

    pitch_type_counts = recent_rows['pitch_type'].value_counts()
    qualifying_pitch_types = pitch_type_counts[pitch_type_counts >= 0.01 * len(recent_rows)].index.tolist()

    pitch_types = (
        recent_rows[recent_rows['pitch_type'].isin(qualifying_pitch_types)]
        .groupby('pitch_type')
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    pitch_type_means = []
    pitch_command = []

    for pitch_type in pitch_types:
        pitch_type_df = recent_rows[recent_rows['pitch_type'] == pitch_type]

        if pitch_type_df.empty:
            continue

        pitch_group_encoded = (
            pitch_type_df.loc[
                (pitch_type_df['pitch_type'] == pitch_type), 
                'PitchGroupEncoded'
            ].mode()[0]
        )

        pitch_type_df = calculate_batter_metrics(pitch_type_df, batter_df)

        ### Issue: No 'VertRelAngle' and 'HorzRelAngle' data
        rel_angles = pitch_type_df[['VertRelAngle', 'HorzRelAngle']].dropna()
        kmeans = KMeans(n_clusters=1, random_state=100)
        kmeans.fit(rel_angles)
        center_vert, center_horz = kmeans.cluster_centers_[0]
        pitch_type_df['DistanceFromCenter'] = np.sqrt(
            (pitch_type_df['VertRelAngle'] - center_vert) ** 2 +
            (pitch_type_df['HorzRelAngle'] - center_horz) ** 2
        )
        command_score = pitch_type_df['DistanceFromCenter'].mean()
        pitch_command.append((pitch_type, command_score))

        expected_features = model.get_booster().feature_names

        pitch_type_df['balls'] = balls
        pitch_type_df['strikes'] = strikes

        rank_df = pitch_type_df[expected_features].copy()
        rank_df['ExpectedRunValue'] = model.predict(rank_df)
        mean_value = rank_df['ExpectedRunValue'].mean()

        if pitch_type in ['FF', 'SI']:
                mean_value -= ((balls - strikes) + 1) * 0.005
            
        pitch_type_means.append((pitch_type, mean_value))

        pitch_type_means_dict = dict(pitch_type_means)
        pitch_command_dict = dict(pitch_command)

        sorted_pitch_types = sorted(pitch_type_means_dict, key=pitch_type_means_dict.get)[:6]

    for i in range(0, len(sorted_pitch_types), 2):
        cols = st.columns(2)

        for col, pitch_type in zip(cols, sorted_pitch_types[i:i+2]):
            with col:
                fig, ax = plt.subplots(figsize=(5, 5))

                if balls == 0 and strikes == 0:
                    synthetic_df = simulate_synthetic_dataframe(recent_rows, batter_df, pitch_type, balls, strikes)
                    synthetic_df['year'] = 2024
                    synthetic_df = calculate_batter_metrics(synthetic_df, batter_df)
                    st.session_state[f'synthetic_data_{pitch_type}'] = synthetic_df
                else:
                    synthetic_df = st.session_state[f'synthetic_data_{pitch_type}']
                expected_features = rv_model.get_booster().feature_names
                model_df = synthetic_df[expected_features]
                model_df['ExpectedRunValue'] = rv_model.predict(model_df)

                count_df = model_df[model_df['CountEncoded'] == 0]
                heatmap_data = (
                    count_df.groupby(['PlateLocHeightBucket', 'PlateLocSideBucket'])['ExpectedRunValue']
                    .mean()
                    .reset_index()
                    .pivot(index="PlateLocHeightBucket", columns="PlateLocSideBucket", values="ExpectedRunValue")
                ).fillna(0)

                mean_value = pitch_type_means_dict.get(pitch_type, 0)

                star_color = "green" if mean_value <= -0.005 else "red"
                stars = next(stars for threshold, stars in star_ratings if mean_value <= threshold)

                st.markdown(
                    f"<h3 style='text-align: center;'>{pitch_type.upper()}</h3>"
                    f"<h4 style='text-align: center; color: {star_color};'>{stars}</h4>",
                    unsafe_allow_html=True
                )

                command_score = pitch_command_dict.get(pitch_type, 1)

                if len(recent_rows[recent_rows['pitch_type'] == pitch_type]) >= 20:
                    sigma = (max(0.25, min(command_score, 2)) * (0.7 + ((balls - strikes) * 0.1)))
                else: 
                    sigma = 0.8 + ((balls - strikes) * 0.1)

                smoothed_weighted_data = gaussian_filter(heatmap_data, sigma=sigma)
                smoothed_weighted_data = smoothed_weighted_data[1:-1, 1:-1]

                sns.heatmap(
                    smoothed_weighted_data,
                    cmap=cmap,
                    xticklabels=False,
                    yticklabels=False,
                    cbar=False,
                    ax=ax
                )

                ax.add_patch(
                    plt.Rectangle((1, 1), 3, 3, edgecolor='black', facecolor='none', linewidth=3)
                )

                ax.invert_yaxis()
                st.pyplot(fig)

    return pitch_types

st.markdown("<h1 style='text-align: center;'>Pitch Calling App</h1>", unsafe_allow_html=True)

if "previous_batter" not in st.session_state:
    st.session_state["previous_batter"] = None
if "previous_pitcher" not in st.session_state:
    st.session_state["previous_pitcher"] = None

def map_ids_to_names(pitches_df, id_list, id_column, name_column):
    """Convert nested ID lists into names based on the most common name in pitches_df."""
    names_list = []
    
    for group in id_list:
        # Filter the dataframe where ID is in the group
        filtered_df = pitches_df[pitches_df[id_column].isin(group)]
        
        if not filtered_df.empty:
            # Find the most common name
            common_name = filtered_df[name_column].mode()
            names_list.append([common_name[0]] if not common_name.empty else [])
        else:
            names_list.append([])  # Empty if no matching rows
    
    return names_list

pitchers = pitches_df['pitcher'].unique()
batters = pitches_df['batter'].unique()

# pitcher_names = map_ids_to_names(pitches_df, pitchers, 'pitcher', 'Pitcher')
# batter_names = map_ids_to_names(pitches_df, batters, 'batter', 'Batter')

# default_pitcher = pitcher_names[0][0] if pitcher_names and pitcher_names[0] else None
# default_batter = batter_names[0][0] if batter_names and batter_names[0] else None

# pitcher_index = next((i for i, group in enumerate(pitcher_names) if default_pitcher in group), 0)
# batter_index = next((i for i, group in enumerate(batter_names) if default_batter in group), 0)

pitcher = st.selectbox("Select Pitcher:", options=pitchers, index=0)
batter = st.selectbox("Select Batter:", options=batters, index=0)

st.write(f"Selected Pitcher: {pitcher}")
st.write(f"Selected Batter: {batter}")

pitcher_df = pitches_df[pitches_df['pitcher'] == pitcher]
batter_df = pitches_df[pitches_df['batter'] == batter]

recent_rows = pitcher_df[pitcher_df['pitch_type'] != 'Undefined'].sort_values(by=['game_date', 'game_pk'], ascending=False).head(500)

if st.session_state["previous_batter"] != batter or st.session_state["previous_pitcher"] != pitcher:
    st.session_state["selected_pitch"] = None
    st.session_state["selected_pitch_call"] = None
    st.session_state["selected_zone"] = None
    st.session_state["previous_batter"] = batter
    st.session_state["previous_pitcher"] = pitcher
    if "prev_pitch" in st.session_state:
        del st.session_state["prev_pitch"]
        
if st.button("Reset All Selections"):
    st.session_state["selected_pitch"] = None
    st.session_state["selected_pitch_call"] = None
    st.session_state["selected_zone"] = None
    if "prev_pitch" in st.session_state:
        del st.session_state["prev_pitch"]

pitch_type_counts = recent_rows['pitch_type'].value_counts()
qualifying_pitch_types = pitch_type_counts[pitch_type_counts >= 0.01 * len(recent_rows)].index.tolist()

pitch_types = (
    recent_rows[recent_rows['pitch_type'].isin(qualifying_pitch_types)]
    .groupby('pitch_type')
    .size()
    .sort_values(ascending=False)
    .index.tolist()
)

sorted_pitch_types = sorted(pitch_types)

st.title("Select a Pitch Type")
columns = st.columns(len(pitch_types))

for i, pitch in enumerate(sorted_pitch_types):
    with columns[i % len(pitch_types)]:
        if st.button(pitch, key=f"pitch_{pitch}"):
            st.session_state["selected_pitch"] = pitch

side_buckets = np.array([-1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8])
height_buckets = np.array([0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2])[::-1]

zone_positions = {
    (i * 7 + j + 1): (side_buckets[j], height_buckets[i]) 
    for i in range(7) for j in range(7)
}

st.title("Pitch Zones (Pitcher POV)")

st.markdown("""
    <style>
        .horizontal-line-top, .horizontal-line-bottom {
            position: absolute;
            left: 26%;
            width: 39%;
            height: 3px;
            background-color: black;
        }
        .horizontal-line-top { margin-top: 145px; }
        .horizontal-line-bottom { margin-top: 335px; }
        .vertical-line-left, .vertical-line-right {
            position: absolute;
            height: 190px;
            background-color: black;
        }
        .vertical-line-left { left: 26%; margin-top: 145px; width: 3px; }
        .vertical-line-right { left: 65%; margin-top: 145px; width: 3px; }
    </style>
    <div class="horizontal-line-top"></div>
    <div class="horizontal-line-bottom"></div>
    <div class="vertical-line-left"></div>
    <div class="vertical-line-right"></div>
""", unsafe_allow_html=True)

columns = st.columns(7, gap="small")
for i in range(7):
    for j in range(7):
        zone_id = i * 7 + j + 1
        with columns[j]:
            if st.button(f"", key=f"zone_{zone_id}"):
                st.session_state["selected_zone"] = zone_id

st.title("Select a Pitch Call")
pitch_calls = ["Ball", "Called Strike", "Swinging Strike", "Foul", "In Play"]
columns = st.columns(len(pitch_calls), gap="large")

for i, pitch_call in enumerate(pitch_calls):
    with columns[i]:
        if st.button(pitch_call, key=f"pitch_call_{pitch_call}"):
            if pitch_call == "In Play":
                st.session_state["selected_pitch"] = None
                st.session_state["selected_pitch_call"] = None
                st.session_state["selected_zone"] = None
                if "prev_pitch" in st.session_state:
                    del st.session_state["prev_pitch"]            
            else:
                st.session_state["selected_pitch_call"] = pitch_call

if st.session_state["selected_pitch"]:
    st.subheader("Selected Pitch Type")
    st.write(f"**Pitch:** `{st.session_state['selected_pitch']}`")

if st.session_state["selected_pitch_call"]:
    st.subheader("Selected Pitch Result")
    st.write(f"**Pitch Call:** `{st.session_state['selected_pitch_call']}`")

if st.session_state["selected_zone"]:
    selected_id = st.session_state["selected_zone"]
    selected_x, selected_y = zone_positions[selected_id]
    st.subheader("Selected Zone Coordinates")
    st.write(f"**Zone Coordinates:** x = {selected_x}, y = {selected_y}")
else:
    st.session_state["selected_zone"] = None

if "prev_pitch" not in st.session_state:
    st.session_state["prev_pitch"] = {}

if st.button("Generate Heatmaps"):
    pitcher_batter_combo = (pitcher, batter)

    if pitcher_batter_combo not in st.session_state["prev_pitch"]:
        st.session_state["prev_pitch"][pitcher_batter_combo] = False
        is_first_time = True
    else:
        st.session_state["prev_pitch"][pitcher_batter_combo] = True
        is_first_time = False

    if "balls" not in st.session_state:
        st.session_state["balls"] = 0
    if "strikes" not in st.session_state:
        st.session_state["strikes"] = 0

    if is_first_time or st.session_state["selected_zone"] is None:
        st.session_state["balls"] = 0
        st.session_state["strikes"] = 0
        st.markdown("<h2 style='text-align: center;'>Count: 0-0</h2>", unsafe_allow_html=True)
        generate_individual_figs(recent_rows, batter_df, rv_model, balls=st.session_state["balls"], strikes=st.session_state["strikes"])
    else:
        recent_rows['prev_pitch'] = True

        recent_rows['prev_pitch_RelSpeed'] = recent_rows[recent_rows['pitch_type'] == st.session_state["selected_pitch"]]['release_speed'].median()
        recent_rows['prev_pitch_HorzBreak'] = recent_rows[recent_rows['pitch_type'] == st.session_state["selected_pitch"]]['pfx_x'].median()
        recent_rows['prev_pitch_InducedVertBreak'] = recent_rows[recent_rows['pitch_type'] == st.session_state["selected_pitch"]]['pfx_z'].median()
        recent_rows['prev_pitch_PlateLocSideBucket'] = selected_x
        recent_rows['prev_pitch_PlateLocHeightBucket'] = selected_y
        recent_rows['prev_pitch_PitchCall'] = st.session_state['selected_pitch_call']
        recent_rows['prev_pitch_PitchType'] = st.session_state['selected_pitch']

        recent_rows['prev_pitch_PitchCall'] = recent_rows['prev_pitch_PitchCall'].apply(
            lambda x: 0 if x == 'Ball' else
            1 if x == 'Called Strike' else
            2 if x == 'Swinging Strike' else
            3 if x == 'Foul' else
            4
        )

        recent_rows['prev_pitch_SamePitch'] = (recent_rows['pitch_type'] == recent_rows['prev_pitch_PitchType']).astype(int)

        prev_pitch_invalid = recent_rows['prev_pitch'].isna() | (recent_rows['prev_pitch'] == False)
        columns_to_update = [
            'prev_pitch_RelSpeed', 'prev_pitch_HorzBreak', 'prev_pitch_InducedVertBreak',
            'prev_pitch_PlateLocSideBucket', 'prev_pitch_PlateLocHeightBucket', 'prev_pitch_PitchCall', 'prev_pitch_SamePitch'
        ]
        recent_rows[columns_to_update] = recent_rows[columns_to_update].where(~prev_pitch_invalid, np.nan)

        selected_pitch_call = st.session_state["selected_pitch_call"]
        if selected_pitch_call == 'Ball':
            st.session_state["balls"] = min(st.session_state["balls"] + 1, 3)
        elif selected_pitch_call in ['Called Strike', 'Swinging Strike', 'Foul']:
            st.session_state["strikes"] = min(st.session_state["strikes"] + 1, 2)

        st.markdown(
            f"<h1 style='text-align: center;'>Count: {st.session_state['balls']}-{st.session_state['strikes']}</h1>",
            unsafe_allow_html=True
        )
        generate_individual_figs(recent_rows, batter_df, prev_pitch_model, balls=st.session_state["balls"], strikes=st.session_state["strikes"])