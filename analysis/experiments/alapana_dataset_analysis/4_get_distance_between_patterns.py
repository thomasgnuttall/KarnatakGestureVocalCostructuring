import pandas as pd
import matplotlib.pyplot as plt
import numpy
import librosa
import os
import fastdtw
import numpy as np
import tqdm
from scipy.signal import savgol_filter
from experiments.alapana_dataset_analysis.dtw import dtw_path, dtw_dtai
from scipy.ndimage import gaussian_filter1d

from experiments.alapana_dataset_analysis.conf import out_dir, root_dir, run_name
from exploration.pitch import get_timeseries, pitch_seq_to_cents,interpolate_below_length
from exploration.io import create_if_not_exists

out_dir = f'{out_dir}/{run_name}/'

r=0.1 # sakoe-chiba radius

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

metadata_path = os.path.join(root_dir, 'data', 'metadata.csv')
metadata = pd.read_csv(metadata_path)
metadata = metadata[~metadata['Audio file'].isnull()]
metadata = metadata[~metadata['Tonic'].isnull()]
tonic_dict = {k:v for k,v in metadata[['Audio file', 'Tonic']].values}

def get_tonic(t, metadata):
    tonic = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Tonic'].values[0]
    return tonic

def get_raga(t, metadata):
    raga = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Raga'].values[0]
    return raga

def get_derivative(pitch, time):

    d_pitch = np.array([((pitch[i+1]-pitch[i])+((pitch[i+2]-pitch[i+1])/2))/2 for i in range(len(pitch)-2)])
    d_time = time[1:-1]

    return d_pitch, d_time

# Iteratively load pitch tracks
pitch_tracks = {}
for t in track_names:
    p_path = f'./data/pitch_tracks/alapana/{t}.csv'
    tonic = get_tonic(t, metadata)
    pitch, time, timestep = get_timeseries(p_path)
    pitch = pitch_seq_to_cents(pitch, tonic=tonic)
    pitch[pitch==None]=0
    pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))
    pitch_d, time_d = get_derivative(pitch, time)
    pitch_tracks[t] = (pitch, time, timestep, pitch_d, time_d)
    #pitch_tracks[t] = (gaussian_filter1d(pitch, 2.5), time, timestep, gaussian_filter1d(pitch_d, 2.5), time_d)

all_patts = pd.read_csv(os.path.join(out_dir, 'all_groups.csv'))
all_patts = all_patts[all_patts['track'].isin(track_names)]
#all_distances = pd.DataFrame(columns=['index1', 'index2', 'path1_start', 'path1_end', 'path2_start', 'path2_end', 'path_length', 'dtw_distance', 'dtw_distance_norm'])

distances_path = os.path.join(out_dir, f'distances.csv')

try:
    print('Removing previous distances file')
    os.remove(distances_path)
except OSError:
    pass
create_if_not_exists(distances_path)


def trim_zeros(pitch, time):
    m = pitch!=0
    i1,i2 = m.argmax(), m.size - m[::-1].argmax()
    return pitch[i1:i2], time[i1:i2]


def smooth(pitch, time, timestep, wms=125):
    pitch2, time2 = trim_zeros(pitch, time)
    wl = round(wms*0.001/timestep)
    wl = wl if not wl%2 == 0 else wl+1
    interp = savgol_filter(pitch2, polyorder=2, window_length=wl, mode='interp')
    return interp, time2


# Iteratively compute dtw distane between each pattern, saving directly to file 

##text=List of strings to be written to file
header = 'index1,index2,pitch_dtw,pitch_dtw_mean,diff_pitch_dtw'
with open(distances_path,'a') as file:
    file.write(header)
    file.write('\n')

    for i, row in tqdm.tqdm(list(all_patts.iterrows())):

        qstart = row.start
        qend = row.end
        qtrack = row.track
        qi = row['index']
        (qpitch, qtime, qtimestep, qpitch_d, qtime_d) = pitch_tracks[qtrack]

        sq1 = int(qstart/qtimestep)
        sq2 = int(qend/qtimestep)
        for j, rrow in all_patts.iterrows():

            rstart = rrow.start
            rend = rrow.end
            rtrack = rrow.track
            rj = rrow['index']
            if qi <= rj:
                continue
            (rpitch, rtime, rtimestep, rpitch_d, rtime_d) = pitch_tracks[rtrack]
            sr1 = int(rstart/rtimestep)
            sr2 = int(rend/rtimestep)

            pat1 = qpitch[sq1:sq2]
            pat1_time = qtime[sq1:sq2]
            pat2 = rpitch[sr1:sr2]
            pat2_time = rtime[sr1:sr2]
            
            pat1[pat1 == None] = 0
            pat2[pat2 == None] = 0
 
            pat1, pat1_time = smooth(pat1, pat1_time, rtimestep)
            pat2, pat2_time = smooth(pat2, pat2_time, qtimestep)
            
            pi = len(pat1)
            pj = len(pat2)
            l_longest = max([pi, pj])
            
            diff1,_ = get_derivative(pat1, pat1_time)
            diff2,_ = get_derivative(pat2, pat2_time)

            diff1, diff1_time = smooth(diff1, pat1_time, rtimestep)
            diff2, diff2_time = smooth(diff2, pat2_time, qtimestep)

            path, dtw_val = dtw_path(pat1, pat2, radius=int(l_longest*r))

            l = len(path)
            dtw_norm = dtw_val/l
            
            path, dtw_val = dtw_path(pat1, pat2, radius=int(l_longest*r), norm=True)

            l = len(path)
            dtw_norm_mean = dtw_val/l

            path, dtw_val = dtw_path(diff1, diff2, radius=int(l_longest*r))
            l = len(path)
            dtw_norm_diff = dtw_val/l

            line =f"{qi},{rj},{dtw_norm},{dtw_norm_mean},{dtw_norm_diff}"
            file.write(line)
            file.write('\n')

