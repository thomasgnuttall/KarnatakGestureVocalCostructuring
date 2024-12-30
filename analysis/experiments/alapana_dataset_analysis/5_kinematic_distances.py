import os

import pandas as pd 
import tqdm
import numpy as np
import fastdtw

import dtaidistance.dtw
from experiments.alapana_dataset_analysis.dtw import dtw_path
from exploration.io import write_pkl, load_pkl
from scipy.signal import savgol_filter

from experiments.alapana_dataset_analysis.conf import out_dir, root_dir, run_name, metadata_path, mocap_dir

r = 0.1 # Sakoe-chiba radius for DTW computations

# mapping of mocap segments to body parts, those marked as importannt are used in this analysis
segment_mapping = {
    'Seg_1':  'Pelvis', # important
    'Seg_2':  'L5',
    'Seg_3':  'L3',
    'Seg_4':  'T12',
    'Seg_5':  'T8',
    'Seg_6':  'Neck',
    'Seg_7':  'Head', # important
    'Seg_8':  'RightShoulder', 
    'Seg_9':  'RightUpperArm',
    'Seg_10':  'RightForearm', # important
    'Seg_11':  'RightHand', # important
    'Seg_12':  'LeftShoulder', 
    'Seg_13':  'LeftUpper Arm',
    'Seg_14':  'LeftForearm', # important
    'Seg_15':  'LeftHand', # important
    'Seg_16':  'RightUpperLeg',
    'Seg_17':  'RightLowerLeg',
    'Seg_18':  'RightFoot',
    'Seg_19':  'RightToe',
    'Seg_20':  'LeftUpperLeg',
    'Seg_21':  'LeftLowerLeg',
    'Seg_22':  'LeftFoot',
    'Seg_23':  'LeftToe'
}

# Functions for mocap normalisation.
def pivot_mocap(df):
    df = df.pivot_table(
        values=['x','y','z'], 
        index=['time', 'time_ms', 'feature'], 
        columns=['segment']).reset_index()

    return df


def force_right_handed(df, is_right):
    columns = list(df.columns)
    feature_columns = [x[1] for x in columns if not x[1]=='']
    index_columns = [x[0] for x in columns if not x[0]=='']
    right_columns = set([x for x in feature_columns if 'Right' in x])
    left_columns = set([x for x in feature_columns if 'Left' in x])
    final_columns = [x.replace('Right','') for x in right_columns]

    if is_right:
        for l in left_columns:
            df = df.drop(l, axis=1, level=1)
    else:
        for r,l in zip(right_columns, left_columns):
            df.loc[:,('x',r)] = -df.loc[:,('x',l)]
            df.loc[:,('y',r)] = df.loc[:,('y',l)]
            df.loc[:,('z',r)] = df.loc[:,('z',l)]
            df = df.drop(l, axis=1, level=1)
    return df

def get_mocap_path(track_name, substring='Acceleration'):
    return [p for p in mocap_paths if track_name in p and substring in p][0]

idict = lambda y: all_groups[all_groups['index']==y].iloc[0].to_dict()

def report_handedness(all_groups):
    n = len(all_groups)
    n_right = round(len(all_groups[all_groups['handedness']=='Right'])*100/n,2)
    n_left = round(len(all_groups[all_groups['handedness']=='Left'])*100/n,2)

    ratio_mean = all_groups['handedness_ratio'].mean()
    ratio_std = all_groups['handedness_ratio'].std()

    n1 = round(sum(all_groups['handedness_ratio']>1)*100/n,2)
    n1_2 = round(sum(all_groups['handedness_ratio']>1.2)*100/n,2)
    n2 = round(sum(all_groups['handedness_ratio']>2)*100/n,2)
    n10 = round(sum(all_groups['handedness_ratio']>10)*100/n,2)
    n100 = round(sum(all_groups['handedness_ratio']>100)*100/n,2)

    print(f'For {n} motifs...')
    print('')
    print(f'{n_left}% are identified as Left handed')
    print(f'{n_right}% are identified as Right handed')
    print('')
    print('The ratio in energy between the identified dominant hand and non-dominant hand has')
    print(f'mean={ratio_mean}')
    print(f'std={ratio_std}')
    print('')
    print(f'The proportion of motifs with a ratio greater than 1 is {n1}%')
    print(f'The proportion of motifs with a ratio greater than 1.2 is {n1_2}%')
    print(f'The proportion of motifs with a ratio greater than 2 is {n2}%')
    print(f'The proportion of motifs with a ratio greater than 10 is {n10}%')
    print(f'The proportion of motifs with a ratio greater than 100 is {n100}%')


def get_motion_energy(i, hand):
    d = idict(i)
    track_name = d['track']
    end = d['end']
    start = d['start']
    mp = get_mocap_path(track_name, substring='Velocity')
    df = mocap[mp]
    min_t = df['time_ms'].iloc[0]
    
    df['time_s'] = df['time_ms'].apply(lambda y: (y-min_t)/1000)

    this_frame = df[(df['time_s']>=start) & (df['time_s']<=end)]

    this_frame = this_frame.loc[:, (slice(None), hand+'Hand')]
    vectors = this_frame.values
    return np.sum(np.apply_along_axis(lambda y: np.linalg.norm(y)**2, 1, vectors))/len(vectors)


def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = 0,0
    px, py, pz = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy, pz


def get_motion_distance(i, j, feature, norm):
    try:
        i_vec = index_features[i][feature]#get_motion_data(i, level, feature, handi, num_dims=num_dims, pelvis=pelvis1, angle=angle1)
        j_vec = index_features[j][feature]#get_motion_data(j, level, feature, handj, num_dims=num_dims, pelvis=pelvis2, angle=angle2)
        
        pi = len(i_vec)
        pj = len(j_vec)
        l_longest = max([pi, pj])

        #dtw_val, path = dtaidistance.dtw_ndim.distance(i_vec, j_vec, window=round(l_longest*0.20), psi=round(l_longest*0.20), use_c=True)
        #dtw_val, path = fastdtw.fastdtw(i_vec, j_vec, dist=None, radius=round(l_longest*0.20))
        path, dtw_val = dtw_path(i_vec, j_vec, radius=round(l_longest*r), norm=norm)

        return dtw_val/len(path)
    except Exception as e:
        raise e
        #return np.nan


def list_dir(path, filetypes=None):
    if filetypes != None:
        return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] in filetypes]
    else:
        return [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames]


def get_mocap_path(track_name, all_paths):
    vel = [x for x in all_paths if track_name in x and 'velocity' in x][0]
    acc = [x for x in all_paths if track_name in x and 'acceleration' in x][0]
    pos = [x for x in all_paths if track_name in x and 'position' in x][0]
    return {'velocity': vel, 'acceleration': acc, 'position': pos}



# Palm, wrist, elbow, shoulder
desirable_segments = ['Seg_1', 'Seg_10', 'Seg_11', 'Seg_14', 'Seg_15', 'Seg_8', 'Seg_12', 'Seg_7', 'Seg_6']

mocap_paths = list_dir(mocap_dir, ['.txt'])

# Iteratively load motion tracks and pivot dataframe to wide format 
columns = ['time', 'time_ms', 'feature', 'segment', 'x', 'y', 'z']
mocap = {}
for mp in tqdm.tqdm(mocap_paths):
    df = pd.read_csv(mp, sep='\t', encoding="ISO-8859-1")
    if len(df) == 0:
        print(f'missing: {mp}')
        continue

    df.columns = columns

    df = df[df['segment'].isin(desirable_segments)]

    df['segment'] = df['segment'].apply(lambda y: segment_mapping[y])

    df = pivot_mocap(df)

    mocap[mp] = df

# Load existing data (motifs and distances)
distances_path = f'{out_dir}/{run_name}/distances.csv'
all_groups_path = f'{out_dir}/{run_name}/all_groups.csv'
gest_path = f'{out_dir}/{run_name}/distances_gestures.csv'

distances = pd.read_csv(distances_path)[['index1', 'index2', 'pitch_dtw','diff_pitch_dtw']] #TODO: ADD 'performance1', 'performer1', 'length1', 'performance2', 'performer2', 'length2', 'length_diff', 
all_groups = pd.read_csv(all_groups_path)

all_groups['LeftEnergy'] = all_groups['index'].apply(lambda y: get_motion_energy(y, 'Left'))
all_groups['RightEnergy'] = all_groups['index'].apply(lambda y: get_motion_energy(y, 'Right'))
all_groups['handedness'] = all_groups.apply(lambda y: 'Right' if y.RightEnergy>y.LeftEnergy else 'Left', axis=1)
all_groups['handedness_ratio'] = all_groups.apply(lambda y: y.RightEnergy/y.LeftEnergy if y.RightEnergy>y.LeftEnergy else y.LeftEnergy/y.RightEnergy, axis=1)
report_handedness(all_groups)

distances = distances.merge(all_groups[['index','handedness']], left_on='index1', right_on='index')
del distances['index']
distances = distances.rename({'handedness':'hand1'}, axis=1)

distances = distances.merge(all_groups[['index','handedness']], left_on='index2', right_on='index')
del distances['index']
distances = distances.rename({'handedness':'hand2'}, axis=1)

# Get Pelvis coordinates
def get_centroid(i, feature, performance=False):
    d = idict(i)
    track_name = d['track']
    end = d['end']
    start = d['start']
    mp = get_mocap_path(track_name, substring='Position')
    df = mocap[mp]
    df = df[1:]
    min_t = df['time_ms'].iloc[0]
    
    df['time_s'] = df['time_ms'].apply(lambda y: (y-min_t)/1000)
    
    if not performance:
        this_frame = df[(df['time_s']>=start) & (df['time_s']<=end)]
    else:
        this_frame = df

    this_frame = this_frame.loc[:, (slice(None), feature)]
    vectors = this_frame.values
    return tuple(vectors.mean(axis=0))



def add_centroid(distances, all_groups, feature, performance):
    all_groups[f'{feature}_centroid'] = all_groups['index'].apply(lambda y: get_centroid(y, feature, performance))
    distances = distances.merge(all_groups[['index',f'{feature}_centroid']], left_on='index1', right_on='index')
    del distances['index']
    distances = distances.rename({f'{feature}_centroid':f'{feature}1'}, axis=1)

    distances = distances.merge(all_groups[['index',f'{feature}_centroid']], left_on='index2', right_on='index')
    del distances['index']
    distances = distances.rename({f'{feature}_centroid':f'{feature}2'}, axis=1)
    return distances

# For each motif compute the spatial center for each body part
distances = add_centroid(distances, all_groups, 'Pelvis', False)
distances = add_centroid(distances, all_groups, 'Neck', True)
distances = add_centroid(distances, all_groups, 'RightShoulder', True)
distances = add_centroid(distances, all_groups, 'LeftShoulder', True)
distances = add_centroid(distances, all_groups, 'Head', False)
distances = add_centroid(distances, all_groups, 'RightHand', False)
distances = add_centroid(distances, all_groups, 'LeftHand', False)


distances2 = distances

# Angle
theta = lambda x0, y0, x1, y1: np.arctan2((y1 - y0), (x1 - x0))
# distance
from scipy.spatial import distance

distances2['angle1'] = distances2.apply(lambda y: theta(y.RightShoulder1[0], y.RightShoulder1[1], y.LeftShoulder1[0], y.LeftShoulder1[1]), axis=1)
distances2['angle2'] = distances2.apply(lambda y: theta(y.RightShoulder2[0], y.RightShoulder2[1], y.LeftShoulder2[0], y.LeftShoulder2[1]), axis=1)

all_groups['angle'] = all_groups.apply(lambda y: theta(y.RightShoulder_centroid[0], y.RightShoulder_centroid[1], y.LeftShoulder_centroid[0], y.LeftShoulder_centroid[1]), axis=1)
all_groups['height'] = all_groups.apply(lambda y: abs(distance.euclidean(y.Head_centroid, y.Pelvis_centroid)), axis=1)

features = [
    ('1daccelerationDTWHand','Acceleration'),
    ('3daccelerationDTWHand','Acceleration'),
    ('1dvelocityDTWHand','Velocity'),
    ('3dvelocityDTWHand','Velocity'),
    ('1dpositionDTWHand','Position'),
    ('3dpositionDTWHand','Position'),
    ('1daccelerationDTWHead','Acceleration'),
    ('3daccelerationDTWHead','Acceleration'),
    ('1dvelocityDTWHead','Velocity'),
    ('3dvelocityDTWHead','Velocity'),
    ('1dpositionDTWHead','Position'),
    ('3dpositionDTWHead','Position')
]

# Normalise Mocap Data
index_features = {}
for i in all_groups['index']:
    index_features[i] = {f[0]:None for f in features}

for i, row in tqdm.tqdm(list(all_groups.iterrows())):
    index = row['index']
    start = row['start']
    end = row['end']
    group = row['group']
    occurrence = row['occurrence']
    track_name = row['track']
    tonic = row['tonic']
    display_name = row['display_name']
    LeftEnergy = row['LeftEnergy']
    RightEnergy = row['RightEnergy']
    handedness = row['handedness']
    handedness_ratio = row['handedness_ratio']
    pelvis = row['pelvis_centroid']
    head = row['Head_centroid']
    HandEnergy = row['HandEnergy']
    pelvis = row['Pelvis_centroid']
    RightShoulder_centroid = row['RightShoulder_centroid']
    LeftShoulder_centroid = row['LeftShoulder_centroid']
    Head_centroid = row['Head_centroid']
    RightHand_centroid = row['RightHand_centroid']
    LeftHand_centroid = row['LeftHand_centroid']
    angle = row['angle']
    height = row['height']
    
    for feature, level in features:

        if 'Head' in feature:
            feature_name = 'Head'
        else:
            feature_name = handedness+'Hand'
        num_dims = int(feature[0])
        d = idict(index)
        mp = get_mocap_path(track_name, substring=level)
        df = mocap[mp]

        min_t = df['time_ms'].iloc[0]

        if level == 'Position' and min_t==0:
            df = df[1:]
            min_t = df['time_ms'].iloc[0]

        df['time_s'] = df['time_ms'].apply(lambda y: (y-min_t)/1000)

        this_frame = df[(df['time_s']>=start) & (df['time_s']<=end)]
        this_frame = this_frame.loc[:, (slice(None), feature_name)]
        
        if level == 'Position':
            this_frame['x'] = this_frame['x'].apply(lambda y: y-pelvis[0])
            this_frame['y'] = this_frame['y'].apply(lambda y: y-pelvis[1])
            this_frame['z'] = this_frame['z'].apply(lambda y: y-pelvis[2])


        this_frame.columns = this_frame.columns.droplevel()
        this_frame.columns = ['x', 'y', 'z']

        this_frame['x'], this_frame['y'], this_frame['z'] = this_frame.apply(lambda y: rotate((y['x'], y['y'], y['z']), -angle), axis=1, result_type='expand').T.values

        if (handedness =='Right') and ('Head' not in feature_name):
            this_frame['x'] = -this_frame['x']

        vectors = this_frame.values
        
        timestep = df['time_ms'].iloc[5]-df['time_ms'].iloc[4]
        wl = round(125/timestep)
        wl = wl if not wl%2 ==0 else wl+1

        if num_dims==1:
            vectors = np.apply_along_axis(np.linalg.norm, 1, vectors)
            po = 2
            vectors = savgol_filter(vectors, polyorder=2, window_length=wl, mode='interp')
        elif num_dims==2:
            vectors = vectors[:,:2]
            vectors[:,0] = savgol_filter(vectors[:,0], polyorder=2, window_length=wl, mode='interp')
            vectors[:,1] = savgol_filter(vectors[:,1], polyorder=2, window_length=wl, mode='interp')
        elif num_dims==3:
            vectors = vectors
            vectors[:,0] = savgol_filter(vectors[:,0], polyorder=2, window_length=wl, mode='interp')
            vectors[:,1] = savgol_filter(vectors[:,1], polyorder=2, window_length=wl, mode='interp')
            vectors[:,2] = savgol_filter(vectors[:,2], polyorder=2, window_length=wl, mode='interp')

        index_features[index][feature] = vectors


index_features_path = f'{out_dir}/{run_name}/index_features.pkl'
write_pkl(index_features, index_features_path)

# Compute pairwise distances
print("acceleration hand")
print('    1d dtw')
distances2['1daccelerationDTWHand'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1daccelerationDTWHand', norm=False), axis=1)
print('    3d dtw')
distances2['3daccelerationDTWHand'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3daccelerationDTWHand', norm=False), axis=1)



print("velocity hand")
print('    1d dtw')
distances2['1dvelocityDTWHand'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1dvelocityDTWHand', norm=False), axis=1)
print('    3d dtw')
distances2['3dvelocityDTWHand'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3dvelocityDTWHand', norm=False), axis=1)



print("position hand")
print('    1d dtw')
distances2['1dpositionDTWHand'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1dpositionDTWHand', norm=False), axis=1)
print('    3d dtw')
distances2['3dpositionDTWHand'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3dpositionDTWHand', norm=False), axis=1)


print("position hand")
print('    1d dtw')
distances2['1dpositionDTWHand_mean'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1dpositionDTWHand', norm=True), axis=1)
print('    3d dtw')
distances2['3dpositionDTWHand_mean'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3dpositionDTWHand', norm=True), axis=1)



print("acceleration Head")
print('    1d dtw')
distances2['1daccelerationDTWHead'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1daccelerationDTWHead', norm=False), axis=1)
print('    3d dtw')
distances2['3daccelerationDTWHead'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3daccelerationDTWHead', norm=False), axis=1)


print("velocity Head")
print('    1d dtw')
distances2['1dvelocityDTWHead'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1dvelocityDTWHead', norm=False), axis=1)
print('    3d dtw')
distances2['3dvelocityDTWHead'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3dvelocityDTWHead', norm=False), axis=1)


print("position Head")
print('    1d dtw')
distances2['1dpositionDTWHead'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'1dpositionDTWHead', norm=True), axis=1)
print('    3d dtw')
distances2['3dpositionDTWHead'] = distances2.apply(lambda y: get_motion_distance(y['index1'], y['index2'],'3dpositionDTWHead', norm=True), axis=1)





# Append info
all_groups['length'] = all_groups['end'] - all_groups['start']

distances2 = distances2.merge(all_groups[['index','length']], left_on='index1', right_on='index')
del distances2['index']
distances2 = distances2.rename({'length':'length1'}, axis=1)

distances2 = distances2.merge(all_groups[['index','display_name']], left_on='index1', right_on='index')
del distances2['index']
distances2 = distances2.rename({'display_name':'performance1'}, axis=1)

distances2 = distances2.merge(all_groups[['index','length']], left_on='index2', right_on='index')
del distances2['index']
distances2 = distances2.rename({'length':'length2'}, axis=1)

distances2 = distances2.merge(all_groups[['index','display_name']], left_on='index2', right_on='index')
del distances2['index']
distances2 = distances2.rename({'display_name':'performance2'}, axis=1)


distances2['performer1'] = distances2['performance1'].apply(lambda y: y.split('_')[0])
distances2['performer2'] = distances2['performance2'].apply(lambda y: y.split('_')[0])

# order cols
distances2 = distances2[[x for x in [
       'index1', 'index2', 'length1', 'length2', 'performer1', 'performer2', 'performance1', 'performance2', 'pitch_dtw', 'diff_pitch_dtw',
       '1dpositionDTWHand', '3dpositionDTWHand',
       '1dvelocityDTWHand', '3dvelocityDTWHand',
       '1daccelerationDTWHand', '3daccelerationDTWHand',
        '1dpositionDTWHead', '3dpositionDTWHead',
       '1dvelocityDTWHead', '3dvelocityDTWHead',
       '1daccelerationDTWHead', '3daccelerationDTWHead'] if x in distances2.columns]]

distances2.to_csv(gest_path, index=False)


