import pandas as pd
from exploration.pitch import get_timeseries, pitch_seq_to_cents,interpolate_below_length
from exploration.io import create_if_not_exists
import pandas as pd
from numba import jit
import numpy
import librosa
import os
import fastdtw
import numpy as np
import tqdm
import dtaidistance.dtw
import soundfile as sf
from experiments.alapana_dataset_analysis.dtw import dtw_path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import librosa
from librosa.feature import spectral_centroid

from experiments.alapana_dataset_analysis.conf import out_dir, root_dir, run_name, metadata_path

r = 0.1
sr = 44100


out_dir = f'{out_dir}/{run_name}/'

metadata_path = 'audio/lara_wim2/info.csv'
metadata = pd.read_csv(metadata_path)
metadata = metadata[~metadata['Audio file'].isnull()]
metadata = metadata[~metadata['Tonic'].isnull()]
tonic_dict = {k:v for k,v in metadata[['Audio file', 'Tonic']].values}

all_groups = pd.read_csv(os.path.join(out_dir, 'all_groups.csv'))

def get_tonic(t, metadata):
    tonic = metadata[metadata['Audio file'].apply(lambda x: x in t)]['Tonic'].values[0]
    return tonic

# get audios and pitches
audio_tracks = {}
pitch_tracks = {}
for t in tqdm.tqdm(all_groups['track'].unique()):
    a_path = f'data/audio/{t}.wav'
    audio_tracks[t], _ = librosa.load(a_path, sr=44100)
    
    p_path = f'/data/pitch_tracks/{t}.csv'
    tonic = get_tonic(t, metadata)
    pitch, time, timestep = get_timeseries(p_path)
    pitch = pitch_seq_to_cents(pitch, tonic)
    pitch[pitch==None]=0
    pitch = interpolate_below_length(pitch, 0, (350*0.001/timestep))
    pitch = pitch.astype(float)
    pitch_tracks[t] = (pitch, time, timestep)


def get_loudness(y, window_size=2048):
    S = librosa.stft(y, n_fft=window_size)**2
    power = np.abs(S)**2
    p_mean = np.sum(power, axis=0, keepdims=True)
    p_ref = np.max(power)
    loudness = librosa.power_to_db(p_mean, ref=p_ref)
    return loudness[0]

@jit(nopython=True)
def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average


def compute_novelty_spectrum(x, Fs=44100, N=1024, H=512, gamma=200.0, M=10, norm=True):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate (Default value = 1)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 256)
        gamma (float): Parameter for logarithmic compression (Default value = 100.0)
        M (int): Size (frames) of local average (Default value = 10)
        norm (bool): Apply max norm (if norm==True) (Default value = True)

    Returns:
        novelty_spectrum (np.ndarray): Energy-based novelty function
        Fs_feature (scalar): Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = H / Fs
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

mel = lambda f: 1/np.log(2) * (np.log(1 + (f/1000))) * 1000

# distance from tonic tracks
dtonic_tracks = {}
# spectral flux tracks
sc_tracks = {}
# loudness tracks
loudness_tracks = {}
for t in all_groups['track'].unique():
    y = audio_tracks[t]
    pitch, time, timestep = pitch_tracks[t]

    # loudness
    loudness = get_loudness(y)
    step = (len(y)/sr)/len(loudness)
    wms = 125
    wl = round(wms*0.001/step)
    wl = wl if not wl%2 == 0 else wl+1
    loudness_smooth = savgol_filter(loudness, polyorder=2, window_length=wl, mode='interp')
    loudness_tracks[t] = (loudness_smooth, step)

    # Spectral centroid
    centroid = spectral_centroid(
        y=y, sr=sr, n_fft=2048, hop_length=512, window='hanning', center=True, pad_mode='constant')[0]

    sc_timestep = (len(y)/sr)/len(centroid)

    wms = 125
    wl = round(wms*0.001/sc_timestep)
    wl = wl if not wl%2 == 0 else wl+1
    centroid125 = savgol_filter(centroid, polyorder=2, window_length=wl, mode='interp')

    centroid125[centroid125<0] = 0
    centroidmel = np.array([mel(f) for f in centroid])

    #spectral_flux, fs_feature = compute_novelty_spectrum(y)
    sc_tracks[t] = (centroidmel, sc_timestep)

    # Distance from tonic
    sameoctave = pitch%1200
    upper = 1200-sameoctave
    stack = np.vstack((upper, sameoctave)).T
    dtonic = stack.min(axis=1)
    if True in np.isnan(dtonic):
        break
    dtonic_tracks[t] = (dtonic, time, timestep)



# window_size = 2048
# loudness = get_loudness(y, window_size)
# step = len(y)/len(loudness)
# loudness_smooth = gaussian_filter1d(loudness, 10)
# loudness_smooth -= loudness_smooth.min()

# s1=176
# s2=186
# sr=44100
# ltrack = loudness_smooth[int(s1*sr/step):int(s2*sr/step)]
# atrack = y[int(s1*sr):int(s2*sr)]
# import matplotlib.pyplot as plt

# plt.plot(list(range(len(ltrack))), ltrack)
# plt.ylabel('Normalised Loudness (dB)')
# plt.xlabel('Time')
# plt.savefig('loudness_smooth.png')
# plt.clf()
# # Write out audio as 24bit PCM WAV
# sf.write('loudness_smooth.wav', atrack, sr, subtype='PCM_24')





audio_distances_path = os.path.join(out_dir, 'audio_distances.csv')

try:
    print('Removing previous distances file')
    os.remove(audio_distances_path)
except OSError:
    pass
create_if_not_exists(audio_distances_path)

##text=List of strings to be written to file
header = 'index1,index2,loudness_dtw,distance_from_tonic_dtw,spectral_centroid'
with open(audio_distances_path,'a') as file:
    file.write(header)
    file.write('\n')

    for i, row in tqdm.tqdm(list(all_groups.iterrows())):

        qstart = row.start
        qend = row.end
        qtrack = row.track
        qindex = row['index']

        (qloudness, qloudnessstep) = loudness_tracks[qtrack]
        (qdtonic, qtime, qtimestep) = dtonic_tracks[qtrack]
        (qsf, qsftimestep) = sc_tracks[qtrack]

        loudness_sq1 = round(qstart/qloudnessstep)
        loudness_sq2 = round(qend/qloudnessstep)
        dtonic_sq1 = round(qstart/qtimestep)
        dtonic_sq2 = round(qend/qtimestep)
        sf_sq1 = round(qstart/qsftimestep)
        sf_sq2 = round(qend/qsftimestep)

        for j, rrow in all_groups.iterrows():
                rstart = rrow.start
                rend = rrow.end
                rtrack = rrow.track
                rindex = rrow['index']
                if qindex <= rindex:
                    continue
                (rloudness, rloudnessstep) = loudness_tracks[rtrack]
                (rdtonic, rtime, rtimestep) = dtonic_tracks[rtrack]
                (rsf, rsftimestep) = sc_tracks[rtrack]

                loudness_sr1 = round(rstart/rloudnessstep)
                loudness_sr2 = round(rend/rloudnessstep)
                dtonic_sr1 = round(rstart/rtimestep)
                dtonic_sr2 = round(rend/rtimestep)
                sf_sr1 = round(rstart/rsftimestep)
                sf_sr2 = round(rend/rsftimestep)

                pat1_loudness = qloudness[loudness_sq1:loudness_sq2]
                pat2_loudness = rloudness[loudness_sr1:loudness_sr2]
                
                pat1_dtonic = qdtonic[dtonic_sq1:dtonic_sq2]
                pat2_dtonic = rdtonic[dtonic_sr1:dtonic_sr2]

                pat1_sf = qsf[sf_sq1:sf_sq2]
                pat2_sf = rsf[sf_sr1:sf_sr2]

                # DTW normal loudness
                p1l = len(pat1_loudness)
                p2l = len(pat2_loudness)

                l_longest = max([p1l, p2l])
                
                path, dtw_val = dtw_path(pat1_loudness, pat2_loudness, radius=r, norm=True)
                l = len(path)
                loudness_dtw = dtw_val/l
                
                # DTW normal dtonic
                p1l = len(pat1_dtonic)
                p2l = len(pat2_dtonic)

                l_longest = max([p1l, p2l])
                path, dtw_val = dtw_path(pat1_dtonic, pat2_dtonic, radius=r)
                l = len(path)
                dtonic_dtw = dtw_val/l

                # DTW Spectral Flux
                p1l = len(pat1_sf)
                p2l = len(pat2_sf)

                l_longest = max([p1l, p2l])
                path, dtw_val = dtw_path(pat1_sf, pat2_sf, radius=r, norm=True)
                l = len(path)
                dsf_dtw = dtw_val/l

                # Write
                line =f"{qindex},{rindex},{loudness_dtw},{dtonic_dtw},{dsf_dtw}"

                file.write(line)
                file.write('\n')



