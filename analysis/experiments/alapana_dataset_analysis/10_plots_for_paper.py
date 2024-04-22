run_name = 'result_0.1'

from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
import seaborn as sns
import tqdm
sns.set_theme()

from scipy.stats import spearmanr
from exploration.io import create_if_not_exists, load_pkl

import seaborn as sns


out_dir = f'{out_dir}/{run_name}/'

results_dir = os.path.join(out_dir, 'analysis', '1_correlations_kinematic_sonic')
results_df_path = os.path.join(results_dir, 'results.csv')

# load data
results = pd.read_csv(results_df_path)

alpha = 0.0001/len(results)

order1 = ['3dpositionDTWHand', '3dvelocityDTWHand', '3daccelerationDTWHand', '3dpositionDTWHead', '3dvelocityDTWHead', '3daccelerationDTWHead']
order2 = ['pitch_dtw', 'diff_pitch_dtw', 'spectral_centroid', 'loudness_dtw']

#old colours slightly lighter
#all_colors = ['#C4F6BB', '#FFDCB9', '#83AFF1', '#FF788A']

# color blind
#all_colors = ['#002051', '#575c6e', '#a49d78', '#fdea45'] # CIVIDIS
#all_colors = ["#440154", "#31688e", "#35b779", "#fde725"] # VIRIDIS

# Color blind lighter
#all_colors = ['#012a69', '#656b80', '#b5ad84', '#faea5c'] # CIVIDIS
all_colors = ["#721787", "#5b8fb5", "#5de3a2", "#ffef61"] # viridis


results['order1'] = results['y'].apply(lambda y: order1.index(y))
results['order2'] = results['x'].apply(lambda y: order2.index(y))

results.sort_values(by=['order2','order1'], inplace=True)

gest_dict = dict(zip(results['y'].unique(),range(results['y'].nunique())))
reverse_gest_dict = {v:k for k,v in gest_dict.items()}

sonic_dict = dict(zip(results['x'].unique(),range(results['x'].nunique())))
reverse_sonic_dict = {v:k for k,v in sonic_dict.items()}

#results['x'] = results['x'].apply(lambda y: sonic_dict[y])
#results['y'] = results['y'].apply(lambda y: gest_dict[y])

results['y'] = results['y'].apply(lambda y: y.replace('DTW', ' ').replace('3d','').replace(' ','\n').lower())
results['x'] = results['x'].apply(lambda y: y.replace('_', ' ').replace('dtw','').replace(' ','\n'))


color_dict = dict(zip(reverse_sonic_dict.keys(), all_colors))
cd_rev = {v:k for k,v in color_dict.items()}


plot_dir = os.path.join(out_dir, 'analysis', 'plots_for_paper')

for level, level_value in [('all', 'all'), ('performer', 'Performer1'), ('performer', 'Performer2'), ('performer', 'Performer3')]:
	color = 'Greens' if level != 'all' else 'YlOrBr'
	if level=='all':
		sns.set_theme(rc={'figure.figsize':(12,3)})
		sns.set(font_scale=0.7)
	else:
		sns.set_theme(rc={'figure.figsize':(6,3)})
		sns.set(font_scale=0.7)
	df = results[(results['level']==level) & (results['level_value']==level_value)]
	df = df[df['p']<alpha]
	df = df[['x', 'y', 'corr']]
	
	df = df.pivot(index="x", columns="y", values="corr")
	df = df[['position\nhand', 'velocity\nhand', 'acceleration\nhand', 'position\nhead', 'velocity\nhead', 'acceleration\nhead']]
	df = df.T
	df = df[['pitch\n', 'diff\npitch\n', 'loudness\n', 'spectral\ncentroid']]
	df.columns = ['f0', 'Δf0', 'loudness\n', 'spectral\ncentroid']
	df = df.T
	plt.close('all')	
	svm = sns.heatmap(df, annot=True, cmap=color, vmin=0, vmax=max(results[(results['p']<0.01) & (results['level']!='performance')]['corr']), annot_kws={"fontsize":12})
	if level=='all':
		plt.title('Spearmans correlation coefficient for all performers', fontsize=10)
	else:
		plt.title(f"Spearmans correlation coefficient for Performer {level_value.replace('Performer','')}", fontsize=10)

	plt.yticks(rotation=0)
	plt.xticks(rotation=0)
	figure = svm.get_figure()
	plt.subplots_adjust(bottom=0.2)
	plt.subplots_adjust(left=0.2)
	plt.xlabel('kinematic feature')
	plt.ylabel('sonic feature')
	plot_path = os.path.join(plot_dir, 'correlations', f'{level}_{level_value}.png')
	figure.savefig(plot_path, dpi=400)

	plt.close('all')
