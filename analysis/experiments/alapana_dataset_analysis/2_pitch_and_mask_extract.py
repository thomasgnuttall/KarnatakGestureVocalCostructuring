import pandas as pd
import numpy as np
import os

# Features Extraction
import csv
from exploration.io import create_if_not_exists
from exploration.pitch import silence_stability_from_file
from compiam import load_model

from experiments.alapana_dataset_analysis.conf import root_dir

# Load pitch extraction model from compIAM package
model = load_model('melody:ftanet-carnatic')

metadata_path = os.path.join(root_dir, 'data', 'metadata.csv')

# Load metadata for tracks
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

# Track names to extract for
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


# Iterate through tracks and extract pitch track and stability mask
all_paths = []
# pitch tracks
failed_tracks = []
for t in track_names:
    print(t)
    print('-'*len(t))
    audio_path = os.path.join(root_dir, f'data/audio/{t}.mp3')
    pitch_path = os.path.join(root_dir, f'data/pitch_tracks/alapana/{t}.csv')
    stab_path = os.path.join(root_dir, f'data/stability_tracks/alapana/{t}.csv')

    tonic = tonic_dict[t]
    raga = get_raga(t, metadata)

    if not os.path.exists(pitch_path):
        create_if_not_exists(pitch_path)
        create_if_not_exists(stab_path)
        
        print('- extracting pitch')
        prediction = model.predict(audio_path)
        pitch = prediction[:,1]
        pitch[np.where(pitch<80)[0]]=0
        prediction[:,1] = pitch

        # pitch track
        with open(pitch_path, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            for row in prediction:
                # write a row to the csv file
                writer.writerow(row)
        
        print('- extracting stability mask')
        silence_stability_from_file(pitch_path, stab_path, tonic=tonic, freq_var_thresh_stab=60, gap_interp=0.350)
    else:
        print(f'{t} already exists!')
    this_path = ((t, raga, tonic), (audio_path, stab_path, pitch_path))
    all_paths.append(this_path)

