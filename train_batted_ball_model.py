import pandas as pd
import helper_functions as hf
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import joblib
from constants import bb_features, bb_target
import warnings
warnings.filterwarnings('ignore')

pitches_df = pd.read_csv('all_pitches.csv')

pitches_df = hf.clean_data(pitches_df, game_only=True)
pitches_df = hf.add_run_value(pitches_df)
pitches_df = hf.bucketize_spray_angle(pitches_df)

batted_ball_df = pitches_df[pitches_df['PitchCall'] == 'InPlay']
batted_ball_df = batted_ball_df.dropna(subset=bb_features + [bb_target])

X = batted_ball_df[bb_features]
y = batted_ball_df[bb_target]

rmse_scorer = make_scorer(mean_squared_error, squared=False)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3), 
        'max_depth': trial.suggest_int('max_depth', 2, 6), 
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),  
        'subsample': trial.suggest_float('subsample', 0.7, 1.0), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }

    model = xgb.XGBRegressor(**params, random_state=100)

    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    cv_scores = cross_val_score(model, X, y, scoring=rmse_scorer, cv=kf)

    return np.mean(cv_scores)

bb_study = optuna.create_study(direction='minimize')
bb_study.optimize(objective, n_trials=10)

bb_best_params = bb_study.best_params
batted_ball_model = xgb.XGBRegressor(**bb_best_params, random_state=100)
batted_ball_model.fit(X, y)

print(f"Best Parameters: {bb_best_params}")

joblib.dump(batted_ball_model, 'batted_ball_model.pkl')

print("Batted Ball Model saved!")