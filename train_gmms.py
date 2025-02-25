import pandas as pd
import joblib
import helper_functions as hf
import warnings
warnings.filterwarnings('ignore')

pitches_df = pd.read_csv('all_pitches.csv')

pitches_df = hf.prepare_data(pitches_df, game_only=True)

gmm_models = hf.fit_gmms(pitches_df)

hf.save_gmm_models(gmm_models, save_dir="gmm_models")
hf.calculate_global_means(pitches_df)
