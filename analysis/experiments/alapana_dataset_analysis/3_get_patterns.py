############# NEW MODEL #############
from experiments.alapana_dataset_analysis.main import main
from experiments.alapana_dataset_analysis.conf import out_dir, root_dir
import faulthandler

from experiments.alapana_dataset_analysis.conf import out_dir, root_dir

faulthandler.enable()
sr = 44100
cqt_window = 1984
s1 = None
s2 = None
gap_interp = 0.35
stab_hop_secs = 0.2
min_stability_length_secs = 1.0
freq_var_thresh_stab = 60
conv_filter_str = 'sobel'
gauss_sigma = None
cont_thresh = 0.15
etc_kernel_size = 10
binop_dim = 3
min_diff_trav = 0.5 #0.1
min_in_group = 2
match_tol = 1
ext_mask_tol = 0.5
n_dtw = 10
thresh_cos = None
top_n = 1000
write_plots = True
write_audio = True
write_patterns = True
write_annotations = False
partial_perc = 0.66
perc_tail = 0.5
plot = False
min_pattern_length_seconds = 1.5

group_len_var = 1 # Current Best: 1
thresh_dtw = 4.5 # Current Best: 8
dupl_perc_overlap_intra = 0.6 # Current Best: 0.6
dupl_perc_overlap_inter = 0.8 # Current Best: 0.75


track_names = [
    'Performer2_Sess1_AnandabhairaviC',
    'Performer3_Sess1_BilahariB',
    'Performer3_Sess2_Bhairavi',
    'Performer2_Sess1_Shankarabharanam',
    'Performer3_Sess1_BilahariA',
    'Performer3_Sess2_Kalyani',
    'Performer2_Sess1_AnandabhairaviA',
    'Performer1_Sess1_Kalyani',
    'Performer2_Sess1_Bhairavi',
    'Performer3_Sess2_Atana',
    'Performer1_Sess1_Atana',
    'Performer1_Sess1_Todi',
    'Performer3_Sess2_Varali',
    'Performer2_Sess1_Kalyani',
    'Performer1_Sess1_Varali',
    'Performer1_Sess2_TodiB',
    'Performer3_Sess1_Todi',
    'Performer1_Sess2_Shankarabharanam',
    'Performer1_Sess2_Kalyani',
    'Performer1_Sess2_TodiA',
    'Performer3_Sess1_Shankarabharanam',
    'Performer1_Sess2_Bilahari',
    'Performer3_Sess1_Kalyani',
    'Performer2_Sess1_Todi',
    'Performer3_Sess2_Anandabhairavi',
    'Performer1_Sess1_Anandabhairavi',
    'Performer2_Sess1_Bilahari',
    'Performer2_Sess1_Varali',
    'Performer2_Sess1_Atana',
    'Performer3_Sess2_Shankarabharanam',
    'Performer1_Sess2_Varali',
    'Performer1_Sess1_Shankarabharanam',
    'Performer1_Sess2_Anandabhairavi',
    'Performer1_Sess1_Bilahari',
    'Performer3_Sess2_Bilahari',
    'Performer3_Sess1_AtanaB',
    'Performer3_Sess1_Bhairavi',
    'Performer3_Sess1_AtanaA',
    'Performer3_Sess2_Sahana',
    'Performer3_Sess1_Varali',
    'Performer3_Sess1_AnandabhairaviA',
    'Performer3_Sess1_AnandabhairaviB',
    'Performer1_Sess2_Atana',
    'Performer3_Sess2_Todi'
]
bin_thresh = 0.045 # parameter tuned on a track basis
total = len(track_names)
bin_thresh_segment = bin_thresh*0.75
main(
    t, run_name, sr, cqt_window, s1, s2,
    gap_interp, stab_hop_secs, min_stability_length_secs,
    60, conv_filter_str, bin_thresh,
    bin_thresh_segment, perc_tail,  gauss_sigma, cont_thresh,
    etc_kernel_size, binop_dim, min_diff_trav, min_pattern_length_seconds,
    min_in_group, match_tol, ext_mask_tol, n_dtw, thresh_dtw, thresh_cos, group_len_var, 
    dupl_perc_overlap_inter, dupl_perc_overlap_intra, None,
    None, partial_perc, top_n, True,
    write_audio, write_patterns, False, plot=False)





############# Transfer to final Folder ###############
######################################################
from distutils.dir_util import copy_tree
from exploration.io import create_if_not_exists

for t in track_names:
    print(f'transferring {t}')
    folder1 = os.path.join(root_dir, f'data/self_similarity/{t}/results/{run_name}/')

    folder2 = os.path.join(out_dir, f'{t}/results/{run_name}/')
    create_if_not_exists(folder2)
    copy_tree(folder1, folder2)


from exploration.io import load_pkl
import pandas as pd

failed_tracks = []
for track_name in track_names:
    starts  = load_pkl(f'{out_dir}/{track_name}/results/{run_name}/starts.pkl')
    lengths = load_pkl(f'{out_dir}/{track_name}/results/{run_name}/lengths.pkl')

    df = pd.DataFrame(columns=['start','end','group'])

    timestep = 0.010000958497076582
    for i,group in enumerate(starts):
        for j,s in enumerate(group):
            l = lengths[i][j]
            s1 = s*timestep
            s2 = (l+s)*timestep
            df = df.append({
                'start':s1,
                'end':s2,
                'group':i,
                'occurrence':j
                }, ignore_index=True)

    df.to_csv(f'{out_dir}/{track_name}/results/{run_name}/groups.csv', index=False)


