# Analysis

In this folder is the code corresponding to the two analyses:

1. Do sonic motif DTW distances covary with spatiotemporal patterns of gesture?
2. Can sonic features be predicted from combined gesture features?

The code for each pipeline step can be found in the scripts in the `experiments/alapana_datasets_analysis` folder.

## 1. Data
It is necessary to download two datasets to use this code, one corresponding to the performance audios and one corresponding to the motion capture data (MOCAP) of the performers movement during performance. Both can be found [here](https://owncloud.gwdg.de/index.php/s/CcTprqZ7dAFIg8Q). 

The data in the Audio folder should be placed in `data/audio` and the data in the Motion Capture folder should be placed in `data/mocap`.

The metadata file should also be downloaded and stored at `data/metadata.csv`.

Please ensure the file `experiments/conf.py` corresponds to these data locations, and if not, update it with the correct relative paths.

## Pipeline Overview
The pipeline consists of 10 scripts:
| Name |Purpose  |
|--|--|
| 2_pitch_and_mask_extract.py | Extract predominant pitch strack and identify sung regions |
| 4_get_distance_between_patterns.py | Compute DTW distance between pitch tracks corresponding to identified melodic motifs |
| 5_kinematic_distances.py |  Process mocap data and compute distances between kinematic motifs co-occurring with melodic motifs|
| 6_audio_distances.py | Extract loudness and spectral centroid time series from audio and compute pairwise DTW distances |
|7_correlations.py |  Correlations for analysis 1|
|8_regression_analysis.py |  Regression models for analysis 2|
|9_plots_for_paper.py  |  Generate plots for analysis 1|
|10_reg_plots_for_paper.py |  Generate plots for analysis 2|
|11_radar_reg_plots_for_paper.py|  Generate plots for analysis 2|


## Individual Scripts

To reproduce the analysis, the the scripts should be ran as follows...

Note; These numbered sections correspond to the scripts in the table above.

### 2. Predominant Pitch Extraction

Extract predominant pitch track using the [compIAM](https://github.com/MTG/compIAM) package. Stable and silent regions of this pitch track are also identified and stored. These are used for the downstream similarity analyses and for excluding uninteresting regions in the pattern finding process.

### 3. Melodic Pattern Finding

The melodic patterns are found for each performance individually using the methodology available as part of the [compIAM](https://github.com/MTG/compIAM) package and presented in...

Thomas Nuttall, Genís Plaja, Lara Pearson, Xavier Serra. “In Search of Sañcaras: Tradition-Informed Repeated Melodic Pattern Recognition in Carnatic Music.” Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022 [[pdf](https://repositori-api.upf.edu/api/core/bitstreams/cca68db1-8203-45d4-8c8c-b2b75c606679/content)]

The original pattern finding model implementation is presented in 

Stefan Lattner, Andreas Arzt, Monika Dörfler. "Learning Complex Basis Functions for Invariant Representations of Audio." Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019. [[pdf](https://arxiv.org/pdf/1907.05982)]

A visual walkthrough of this pipeline step can found in the compIAM documentation [here](https://mtg.github.io/IAM-tutorial-ismir22/melodic_analysis/melodic-pattern-discovery.html).

Once all patterns are found across all performances. They are stored in a dataframe with columns:

| Column Name | Contents  |
|--|--|
| index | unique index corresponding to this pattern |
| start | start time in seconds|
| end | end time in seconds|
| display_name | name of track in which pattern is found |

This dataframe should be stored at `data/all_groups.csv` before running subsequent scripts.

### 4. Get Distance Between Patterns

Compute the DTW distance between each pairwise combination of automatically identified melodic patterns. The dynamic time warping implementation is found in `dtw.py:dtw_path`. 

### 5. Kinematic Distances

Load the mocap data and normalise. For each region of the kinematic track that corresponds to the melodic motifs identified in the previous sections. Position, velocity, and acceleration tracks for the hand and head are normalised such that the performers are all facing in the same direction with respect to the origin and such that all motifs have the same "handedness".

### 6. Audio Distances

For each melodic motif exctract the loudness and spectral centroid time series. Normalise and compute the pairwise distance between these tracks. The output of this script is the final distances dataframe upon which all analyses are performed.

### 7. Correlations

Analysis 1 script.

### 8. Regression Analysis

Analysis 2 script.

### 9. Plots for Paper

Generate correlation plots for Analysis 1

### 10. Regression Plots for Paper

Generate regression plots for Analysis 2

### 11. Radar Plots

Generate radar plots for Analysis 2
