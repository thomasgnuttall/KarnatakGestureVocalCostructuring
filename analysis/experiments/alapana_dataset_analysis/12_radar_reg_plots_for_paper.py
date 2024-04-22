run_name = 'result_0.1'
from math import pi
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

from experiments.alapana_dataset_analysis.conf import out_dir, root_dir, run_name, metadata_path, mocap_dir

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

results['target'] = results['target'].apply(lambda y: y.replace('pitch_dtw','f0').replace('diff_','Δ'))
order1 = ['hand', 'head', 'all']
order2 = ['f0', 'Δf0', 'spectral_centroid', 'loudness_dtw']

results['order1'] = results['body_part'].apply(lambda y: order1.index(y))
results['order2'] = results['target'].apply(lambda y: order2.index(y))

results.sort_values(by=['order2','order1'], inplace=True)

results['target'] = results['target'].apply(lambda y: y.replace('_', ' ').replace('dtw','').replace(' ','\n'))

plot_dir = os.path.join(out_dir, 'analysis', 'plots_for_paper')

results['test_score'] = results['test_score'].apply(lambda y: max([y,0]))
for level, level_value in [('all', '0'), ('performer', 'Performer1'), ('performer', 'Performer2'), ('performer', 'Performer3')]:
    
    if level=='all':
        all_colors = ['#012a69', '#656b80', '#b5ad84', '#faea5c']
    else:
        all_colors = ["#721787", "#5b8fb5", "#5de3a2", "#ffef61"]
    df = results[(results['level']==level) & (results['level_value']==level_value)]
    df = df[['target', 'body_part', 'test_score']]
    df = df.pivot(index=['body_part'], columns=['target']).reset_index()
    df.columns = df.columns.droplevel()
    df.columns = ['body_part'] + list(df.columns[1:])

    df = df[['body_part','loudness\n','Δf0','spectral\ncentroid','f0']]

    plt.close('all')
    # ------- PART 1: Create background
    plt.figure(figsize=(5,5))
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi +pi/4 for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    ax.tick_params(pad=14) 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
     
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=14)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3], ["0.1","0.2","0.3"], color="darkgrey", fontsize=18)
    plt.ylim((0,0.35))

    # ------- PART 2: Add plots
     
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 body_parts makes the chart unreadable
     
    # Ind1
    this = df.loc[0]
    name = this.body_part
    values=this.drop('body_part').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='r', linewidth=0.75, linestyle='solid', label=name)
    ax.fill(angles, values, 'r', alpha=0.1)
     
    # Ind2
    this = df.loc[1]
    name = this.body_part
    values=this.drop('body_part').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='g', linewidth=0.75, linestyle='solid', label=name)
    ax.fill(angles, values, 'g', alpha=0.1)

    # Ind2
    this = df.loc[2]
    name = this.body_part
    values=this.drop('body_part').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='b',linewidth=0.75, linestyle='solid', label=name)
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Add legend
    plt.legend(loc='center left')
    if level=='all':
        plt.title('All performers', fontsize=20, pad=12)
    else:
        plt.title(f'{level_value.replace("Performer","Performer ")}', fontsize=20, pad=12)

    plot_path = os.path.join(plot_dir, 'regressions', f'radar_{level}_{level_value}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    # Show the graph
    plt.close('all')







