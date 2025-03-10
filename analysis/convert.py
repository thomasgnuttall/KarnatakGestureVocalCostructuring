#!/usr/bin/env python
"""
Created on July 05, 2019

@author: Stefan Lattner

Sony CSL Paris, France

"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.signal import convolve2d
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from complex_auto.cqt import to_cqt_repr, standardize
from complex_auto.complex import Complex
from complex_auto.config import get_config, config_to_args
from complex_auto.plot import plot_fs_2d
from complex_auto.transform import rPCA
from complex_auto.util import save_pyc_bz, \
    cuda_variable, to_numpy, prepare_audio_inputs

from exploration.pitch import silence_stability_from_file

LOGGER = logging.getLogger(__name__)


def evaluate(loader, cuda, fn="pca_trans.png"):
    angles_diff = []
    instance_nrs = []
    labels = []
    mags_all_x = []
    mags_all_y = []
    model.cpu()
    model.eval()
    for batch_idx, (x, y, transform, instance_nr, distance) in enumerate(
            loader):
        mags_x, angles_x = model(x)
        mags_y, angles_y = model(y)
        mags_all_x.append(mags_x.detach())
        mags_all_y.append(mags_y.detach())
        angles_diff.append((angles_x - angles_y).detach())
        instance_nrs.append(instance_nr)
        labels.append(distance)
        if batch_idx == 50:
            break

    angles = np.vstack(angles_diff)

    mags_all_x = np.vstack(mags_all_x)
    mags_all_y = np.vstack(mags_all_y)
    instance_nrs = np.hstack(instance_nrs)
    distances = ((np.hstack(labels) * 100) // 10) / 10

    mags_prod = mags_all_x * mags_all_y
    quartile = mags_prod.copy()
    np.matrix.sort(quartile)
    quartile = quartile[:, 0]

    angles[mags_prod < quartile[:, None]] = 0
    angles = np.arcsin(np.sin(angles))

    dims = 6
    pca = rPCA(angles, M=dims)
    vals = pca.pca_transform()
    print(f"Var explained = {pca.pca_expvar()}")
    print(f"size_vals = {vals.shape}")
    for i in range(dims-2):
        vals_dim = vals[:, i:i+2]
        plot_fs_2d(vals_dim, distances, save_fn=os.path.join(out_dir,
                                                         f"dim{i}"+fn))
    if cuda:
        model.cuda()


def get_balanced(data_eval_, nr_nn_):
    x = []
    y = []
    classes = []
    for j in range(10):
        n = 0
        k = 0
        while n < nr_nn_ // 10:
            if data_eval_[k, 3] == j:
                x.append(data_eval_[k, 0])
                y.append(data_eval_[k, 1])
                classes.append(j)
                n += 1
            k += 1
    x = np.vstack(x)
    y = np.vstack(y)
    return x, y, classes


def classify(mags, mags_test, classes, classes_test, type='knn'):
    if type == 'knn':
        metric = 'cosine'
        neigh = KNeighborsClassifier(n_neighbors=10, metric=metric)
        neigh.fit(mags, classes)
        preds = neigh.predict(mags_test)
        error = 1 - (np.sum(preds == np.array(classes_test)) / len(preds))
        return error
    else:
        logreg = LogisticRegression(solver='newton-cg', max_iter=200)
        logreg.fit(mags, classes)
        preds = logreg.predict(mags_test)
        error = 1 - (np.sum(preds == np.array(classes_test)) / len(preds))
        return error


def create_ss_matrix(ampls, mode='cosine'):
    matrix = squareform(pdist(np.vstack(to_numpy(ampls)),
                              metric=mode))
    return matrix


def to_amp_phase(model, input, step_size, length_ngram, cuda):
    if cuda:
        model.cuda()
    model.eval()
    ngrams = []
    for i in range(0, len(input) - length_ngram, step_size):
        curr_ngram = input[i:i + length_ngram].reshape((-1,))
        curr_ngram = standardize(curr_ngram)
        ngrams.append(curr_ngram)

    x = cuda_variable(torch.FloatTensor(np.vstack(ngrams)))

    ampl, phase = model(x)
    return ampl, phase


def create_matrices(dir_results, model, files, length_ngram, cuda, data_type, n_bins, bins_per_oct, fmin, hop_length, step_size=1, mode='cosine'):
    out_files = []
    for file in files:
        print(f"Creating self-similarity matrix from features of {file}..")
        data = get_input_repr(file, data_type, n_bins, bins_per_oct, fmin, hop_length)
        ampls, phases = to_amp_phase(model, data, step_size=step_size, length_ngram=length_ngram, cuda=cuda)

        matrix = create_ss_matrix(ampls, mode=mode)
        matrix = np.pad(matrix, ((0, 9), (0, 9)), mode='constant',
                        constant_values=matrix.max())
        matrix = 1 / (matrix + 1e-6)

        for k in range(-8, 9):
            eye = 1 - np.eye(*matrix.shape, k=k)
            matrix = matrix * eye

        flength = 10
        ey = np.eye(flength) + np.eye(flength, k=1) + np.eye(flength, k=-1)
        matrix = convolve2d(matrix, ey, mode="same")
        matrix -= matrix.min()
        matrix /= (matrix.max() + 1e-8)

        fn_base = os.path.basename(file)
        out_npy = os.path.join(dir_results, f"{fn_base}.npy")
        out_plot = os.path.join(out_dir, f"{fn_base}.png")
        np.save(out_npy, matrix)
        out_files.append(out_npy)
        print(f"Written matrix to {out_npy}")
        print(f"SS Matrix shape = {matrix.shape}")
        plt.clf()
        plt.imsave(out_plot, matrix, cmap="hot")
        print(f"Saved matrix plot to {out_plot}\n")

    with open(os.path.join(out_dir, "ss_matrices_filelist.txt"), "w+") as f:
        for file in out_files:
            f.write(file + "\n")


def get_input_repr(file, data_type, n_bins, bins_per_oct, fmin, hop_length):
    if data_type == "cqt":
        sr = 44100
        print(f"Transforming to CQT: {file}, sample rate = {sr}")
        repr = to_cqt_repr(file, n_bins, bins_per_oct, fmin,
                          hop_length, use_nr_samples=-1, sr=sr,
                          standard=True, mult=1.)
        repr = repr.transpose()

    return repr


def save_repres(dir_results, model, files, length_ngram, cuda, data_type, n_bins, bins_per_oct, fmin, hop_length, step_size=1):
    for file in files:
        print(f"Calculating features of {file}..")
        data = get_input_repr(file, data_type, n_bins, bins_per_oct, fmin, hop_length)
        amp, phase = to_amp_phase(model, data, step_size=step_size, length_ngram=length_ngram, cuda=cuda)
        results = np.array([amp, phase])
        out_file = os.path.join(dir_results, os.path.basename(file) +
                                "_repres.pyc.bz")
        print(f"Saving features to {out_file}..")
        save_pyc_bz(results, out_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a complex '
                                                 'autoencoder on different '
                                                 'types of data')

    parser.add_argument('run_keyword', type=str, default="experiment1",
                        help='keyword used for output path')
    parser.add_argument('input_files', type=str, default="",
                        help='text file containing a list of text files '
                             'each containing a list of audio files for '
                             'training input')
    parser.add_argument('config', type=str, default="config.ini",
                        help='config file using config_spec.ini as spec')
    parser.add_argument('--step-size', type=int, default=1,
                        help='step size for sliding window to calculate '
                             'representations')
    parser.add_argument('--refresh-cache', action="store_true", default=False,
                        help='reload and preprocess data')
    parser.add_argument('--self-sim-matrix', action="store_true",
                        default=False, help='reload and preprocess data')

    args = parser.parse_args()

    config = get_config(args.config, 'config_spec.cfg')
    ARGS = config_to_args(config, args)

    ARGS.cuda = torch.cuda.is_available()

    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    if ARGS.cuda:
        torch.cuda.manual_seed(ARGS.seed)

    kwargs = {'num_workers': 0, 'pin_memory': True} if ARGS.cuda else {}
    out_dir = os.path.join("output", ARGS.run_keyword)

    assert os.path.exists(out_dir), f"The output directory {out_dir} does " \
        f"not exist. Did you forget to train using the run_keyword " \
        f"{ARGS.run_keyword}?"

    if not os.path.exists(ARGS.cache_dir):
            os.makedirs(ARGS.cache_dir)

    if ARGS.data_type == 'cqt':
        in_size = ARGS.n_bins * ARGS.length_ngram
    else:
        raise AttributeError(f"Data_type {ARGS.data_type} not supported. "
                             f"Possible type is 'cqt'.")
    model = Complex(in_size, ARGS.n_bases, dropout=ARGS.dropout)
    model_save_fn = os.path.join(out_dir, "model_complex_auto_"
                                         f"{ARGS.data_type}.save")
    model.load_state_dict(torch.load(model_save_fn, map_location='cpu'), strict=False)

    if ARGS.cuda:
        model.cuda()

    if ARGS.data_type == 'cqt':
        files = prepare_audio_inputs(ARGS.input_files)
        mode = 'cosine'
    else:
        raise AttributeError(f"Data_type {ARGS.data_type} not supported. "
                             f"Possible type is 'cqt'.")

    save_repres(out_dir, model, files, ARGS.length_ngram, ARGS.cuda, ARGS.data_type, ARGS.n_bins, ARGS.bins_per_oct, ARGS.fmin, ARGS.hop_length, step_size=ARGS.step_size)

    if ARGS.self_sim_matrix:
        create_matrices(out_dir, model, files, ARGS.length_ngram, ARGS.cuda, ARGS.data_type, ARGS.n_bins, ARGS.bins_per_oct, ARGS.fmin, ARGS.hop_length, step_size=ARGS.step_size)


