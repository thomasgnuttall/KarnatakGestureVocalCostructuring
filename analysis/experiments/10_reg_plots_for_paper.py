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

results_dir = os.path.join(out_dir, 'analysis', '2_regression_kinematic_sonic')
head_df_path = os.path.join(results_dir, 'predicting_sonic_results_head.csv')
hand_df_path = os.path.join(results_dir, 'predicting_sonic_results_hand.csv')
all_df_path = os.path.join(results_dir, 'predicting_sonic_results.csv')


# load data
results_head = pd.read_csv(head_df_path)
results_head['body_part'] = 'head'
results_hand = pd.read_csv(hand_df_path)
results_hand['body_part'] = 'hand'
results_all = pd.read_csv(all_df_path)
results_all['body_part'] = 'all'

results = pd.concat([results_head, results_hand, results_all])
results = results[results['level']!='performance']

order1 = ['hand', 'head', 'all']
order2 = ['pitch_dtw', 'diff_pitch_dtw', 'spectral_centroid', 'loudness_dtw']

results['order1'] = results['body_part'].apply(lambda y: order1.index(y))
results['order2'] = results['target'].apply(lambda y: order2.index(y))

results.sort_values(by=['order2','order1'], inplace=True)

results['target'] = results['target'].apply(lambda y: y.replace('_', ' ').replace('dtw','').replace(' ','\n'))

plot_dir = os.path.join(out_dir, 'analysis', 'plots_for_paper')

for level, level_value in [('all', '0'), ('performer', 'Performer1'), ('performer', 'Performer2'), ('performer', 'Performer3')]:
    df = results[(results['level']==level) & (results['level_value']==level_value)]
    if level=='all':
        all_colors = ['#012a69', '#656b80', '#b5ad84', '#faea5c']
    else:
        all_colors = ["#721787", "#5b8fb5", "#5de3a2", "#ffef61"]
    # set width of bar 
    barWidth = 0.2
    plt.close('all')
    fig = plt.subplots(figsize = (10, 4))
    
    # set height of bar 
    hand = df[df['body_part']=='hand']['test_score'].values
    head = df[df['body_part']=='head']['test_score'].values
    both = df[df['body_part']=='all']['test_score'].values
     
    # Set position of bar on X axis 
    br1 = np.arange(len(head))
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
     
    # Make the plot
    plt.bar(br1, hand, color =all_colors[0], width = barWidth, label ='hand') 
    plt.bar(br2, head, color =all_colors[1], width = barWidth, label ='head') 
    plt.bar(br3, both, color =all_colors[2], width = barWidth, label ='both') 

    plt.xlabel('sonic target', fontsize=12)
    plt.ylabel('test R$^2$', fontsize=12)
    plt.ylim(0,0.35)
    plt.xticks([r + barWidth for r in range(len(head))], [x.replace('_', ' ').replace('dtw','') for x in order2], fontsize=12)
 
    plt.legend(loc='upper right')
    if level=='all':
        plt.title('Predicting sonic target from kinematic features across all performers', fontsize=14)
    else:
        plt.title(f'Predicting sonic target from kinematic features for Performer {level_value.replace("Performer","")}', fontsize=14)

    plt.subplots_adjust(bottom=0.2)
    plot_path = os.path.join(plot_dir, 'regressions', f'{level}_{level_value}.png')
    plt.savefig(plot_path)
    plt.close('all')


