# Analysis

In this folder is the code corresponding to the two analyses:

1. Do sonic motif DTW distances covary with spatiotemporal patterns of gesture?
2. Can sonic features be predicted from combined gesture features?

The code for each pipeline step can be found in the scripts in the `experiments` folder.

It is necessary to download two datasets to use this code, one corresponding to the performance audios (\<link removed for blind review>), and one corresponding to the motion capture data (MOCAP) of the performers movement during performance (\<link removed for blind review>). Please update the file `experiments/conf.py` with the path corresponding to this data. 

## Pipeline
The pipeline consists of 12 scripts:
| Name |Purpose  |
|--|--|
| 1_pitch_and_mask_extract.py | Extract predominant pitch strack and identify sung regions |
| 2_CAE_feature_extraction.py | Encode audio using auto encoder and compute self similarity for motif finding|
| 3_get_patterns.py |  Extract repeated melodic motifs from self similarity matrices|
| 4_get_distance_between_patterns.py | Compute DTW distance between pitch tracks corresponding to identified melodic motifs |
| 5_acceleration_analysis.py |  Process mocap data and compute distances between kinematic motifs co-occurring with melodic motifs|
| 6_pitch_features.py | Extract loudness and spectral centroid time series from audio |
|7_correlations.py |  Correlations for analysis 1|
|8_regression_analysis.py |  Regression models for analysis 2|
|9_classification_analysis.py|  Classification models for analysis 2|
|10_plots_for_paper.py  |  Generate plots for analysis 1|
|11_reg_plots_for_paper.py |  Generate plots for analysis 2|
|12_radar_reg_plots_for_paper.py|  Generate plots for analysis 2|


