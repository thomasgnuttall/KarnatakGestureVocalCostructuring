run_name = 'result_0.1'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

sns.set_theme()

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists
from exploration.visualisation import flush_matplotlib

from experiments.alapana_dataset_analysis.conf import out_dir, root_dir, run_name, metadata_path, mocap_dir

out_dir = f'{out_dir}/{run_name}/'
distances_path = os.path.join(out_dir, 'distances_gestures.csv')
all_groups_path = os.path.join(out_dir, 'all_groups.csv')
audio_features_path = os.path.join(out_dir, 'audio_distances.csv')

# load data
distances = pd.read_csv(distances_path)
all_groups = pd.read_csv(all_groups_path)
audio_features = pd.read_csv(audio_features_path)

pitch_targets = ['pitch_dtw', 'diff_pitch_dtw']

audio_targets = ['loudness_dtw',  'spectral_centroid']

features = [
       '3dpositionDTWHead', '3dvelocityDTWHead', '3daccelerationDTWHead',
       '3dpositionDTWHand', '3dvelocityDTWHand', '3daccelerationDTWHand']

audio_distance = distances.merge(audio_features, on=['index1', 'index2'])

targets = features
features = pitch_targets + audio_targets


## Regression
levels = ['all', 'performer', 'performance']

scoring='r2'
results = pd.DataFrame(columns=['target', 'level', 'level_value', 'test_score', 'train_score', 'best_params', 'n_samples'])
for target in targets:
  print(target)
  for level in levels:
      print(level)
      cols = [x for x in audio_distance.columns if level in x]
      uniqs = set()
      for c in cols:
          uniqs = uniqs.union(set(audio_distance[c].unique()))
      if level == 'all':
          uniqs = [0]
      for u in uniqs:
          if level == 'all':
              this_df = audio_distance
          else:
              this_df = audio_distance[(audio_distance[cols[0]]==u) & (audio_distance[cols[1]]==u)]           
          if len(this_df) > 100:
                X = this_df[features].values
                y = this_df[target].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Predictor has many parameters, we want to select the best so we train 
                # a model for each combination of the following parameters
                # selecting the best as it is evaluated on our training set
                all_params = {
                    "n_estimators": [50, 100, 150, 200],
                    'learning_rate':[0.001, 0.01, 0.1],
                    'max_depth': [2, 4, 8, 10]
                }

                # Initialise predictor
                gbc = GradientBoostingRegressor()

                # set up gridsearch (will train one classifier for each combination of parameters)
                clf = GridSearchCV(gbc, all_params, cv=3, scoring=scoring)

                # fit for just good features (might take some time)
                clf.fit(X_train, y_train)
                test_score = clf.score(X_test, y_test)
                train_score = clf.best_score_

                to_append = {
                    'target': target,
                    'level': level,
                    'level_value': u,
                    'test_score': test_score,
                    'train_score': train_score,
                    'best_params': clf.best_params_,
                    'n_samples': len(this_df)
                }
                results = results.append(to_append, ignore_index=True)

results_df_path = os.path.join(out_dir, 'analysis', '3_regression_kinematic_kinematic', f'predicting_kinematic_results.csv')
create_if_not_exists(results_df_path)
results.to_csv(results_df_path, index=False)

