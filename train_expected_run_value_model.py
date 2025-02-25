import pandas as pd
import joblib
import helper_functions as hf
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
from constants import rv_features, rv_target
import warnings
warnings.filterwarnings('ignore')

pitches_df = pd.read_csv('all_pitches.csv')
global_means = pd.read_csv('global_means.csv')

pitches_df = hf.prepare_data(pitches_df, game_only=True)

pitches_df = add_probabilities(pitches_df)
pitches_df, pivoted_values = hf.calculate_shrunken_means(pitches_df, global_means)
pitches_df = hf.compute_batter_stuff_value(pitches_df, pivoted_values)

pitches_df = pitches_df.dropna(subset=rv_features + [rv_target])

X = pitches_df[rv_features]
y = pitches_df[rv_target]

rmse_scorer = make_scorer(mean_squared_error, squared=False)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500)
    }

    model = xgb.XGBRegressor(**params, random_state=100)

    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    cv_scores = cross_val_score(model, X, y, scoring=rmse_scorer, cv=kf)

    return np.mean(cv_scores)

rv_study = optuna.create_study(direction='minimize')
rv_study.optimize(objective, n_trials=5)

rv_best_params = rv_study.best_params
rv_model = xgb.XGBRegressor(**rv_best_params, random_state=100)
rv_model.fit(X, y)

print(f"Best Parameters: {rv_best_params}")

joblib.dump(rv_model, 'rv_model.pkl')

print("Expected Run Value Model saved!")