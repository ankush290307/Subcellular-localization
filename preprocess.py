#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import os
import csv
import subprocess


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__author__ = 'Alejandro Fontal'


def get_1h_dict(aa_string, props_file, add_props=True):
    """
    Given a string of unique characters, generates dictionary of 1-hot vectors
    with characters as keys and vectors as values.

    """

    aa_dict = {}

    for idx, aa in enumerate(aa_string):

        if idx > 0:
            aa_dict[aa] = np.zeros(idx-1).tolist() + [1] +\
                          np.zeros(len(aa_string)-idx).tolist()
        else:
            aa_dict[aa] = [1] + np.zeros(len(aa_string)-1).tolist()

    aa_dict["X"] = np.zeros(len(aa_string)).tolist()

    if add_props:
        with open(props_file) as csvfile:
            aa_props = csv.reader(csvfile)
            for idx, row in enumerate(aa_props):
                if idx > 0:
                    aa = row[0]
                    props = row[1:]
                    aa_dict[aa] += map(float, props)

    return aa_dict


def seq2onehot(seq, aa_dict):
    """
    Takes a sequence(string) of length n and a dictionary with m keys and
    converts it to a 2 dimensional vector of nxm.

    """
    onehot = []

    # Convert ambiguous amino acids into actual ones.
    for aa in seq:

        if aa == "Z":
            aa = "Q"
        elif aa == "B":
            aa = "R"
        elif aa == "J":
            aa = "L"

        onehot += [aa_dict[aa]]

    return onehot


def fasta_parse(fasta_fn):
    """
    :param fasta_fn: Filename of the FASTA file to parse
    :return: A list containing the sequences in the FASTA file.
    """
    with open(fasta_fn) as fasta_file:
        fasta_list = fasta_file.read().splitlines()

        parsed_seqs = []
        for line in fasta_list:
            if not line.startswith(">"):
                parsed_seqs.append(line)

    return parsed_seqs


def seq_process(seqs, seq_len):
    """

    :param seqs: List containing protein sequences
    :param seq_len: Length of the final processed sequences (integer)
    :return: List of processed sequences of length= seq_len. Filled with Xs
    """

    processed_seqs = []

    for seq in seqs:
        l = len(seq)

        if l > seq_len:
            # Keep the N- and C- Terminal ends
            half = seq_len // 2
            res = seq_len % 2
            left = seq[:half + res]
            right = seq[l - half:]
            processed_seqs.append(left + right)

        else:
            # Add X's in the middle, keeping the ends.
            half = l // 2
            res = l % 2
            left = seq[0:half + res]
            middle = "X" * (seq_len - l)
            right = seq[l - half:]
            processed_seqs.append(left + middle + right)

    return processed_seqs


def get_pssm_dict(filters_tensor, n_filters, aa_string, aa_vec_len=20):
    """

    :param filters_tensor: tensor containing the weights of the conv filters
    :param n_filters: number of filters applied in the conv layer
    :param aa_vec_len: total number of features
    :param aa_string: String containing the aa in the same order as tf input
    :return: dictionary with a df with the pssm for each of the filters
    """
    pssm_dict = {}
    for fn in range(n_filters):    # for each of the filters
        filt_matrix = np.zeros([len(filters_tensor), aa_vec_len])
        for aa in range(aa_vec_len):    # for each of the features
            for pos in range(len(filters_tensor)):    # for each filter pos.
                # fill matrix with raw values
                filt_matrix[pos, aa] = filters_tensor[pos][aa][0][fn]
        # Check which column scores the most, and reshape to make it 1
        # This way we can check relative importance on a scale from -1 to 1
        filt_matrix = np.divide(filt_matrix, np.abs(filt_matrix).sum(axis=1).max())
        # Format to generate pd df exportable to .txt and usable by seq2logo
        filt_df = pd.DataFrame(filt_matrix)
        filt_df.columns = list(aa_string)
        # Store in dictionary object
        pssm_dict["filter " + str(fn + 1)] = filt_df

    return pssm_dict

def run_seq2logo(input_fn, output_fn):

    cmd = "../../../seq2logo-2.0/Seq2Logo.py -f " + r'"{}"'.format(input_fn) +\
          " -o {}".format(output_fn) + " -I 5 -u " + \
          r'"Relative Importance" -S 1 -C 0'

    try:
        subprocess.check_call(cmd, shell=True)
        return output_fn

    except subprocess.CalledProcessError:
        print ("Seq2logo encountered an error during execution")


class DataSet:
    def __init__(self, seqdir, props_file, add_props=True, seq_len=1000,
                 flatten=False):

        self.seqfiles = os.listdir(seqdir)
        self.aa_string = "ARNDCEQGHILKMFPSTWYV"  # 20 aa
        self.aa_dict = get_1h_dict(self.aa_string, props_file,
                                   add_props=add_props)
        self.labels = [i.replace(".fasta", "") for i in self.seqfiles]
        self.raw_seqs = [fasta_parse(seqdir + x) for x in self.seqfiles]
        self.all_seqs = [seq_process(x, seq_len) for x in self.raw_seqs]
        self.train_seqs = []
        self.test_seqs = []
        self.train_tensor = []
        self.test_tensor = []

        for seqs in self.all_seqs:  # for seqs belonging to one specific label
            train_n = int(round(0.8 * len(seqs), 0))
            train_idxs = np.random.choice(len(seqs), train_n, replace=False)
            test_idxs = list(set(range(len(seqs))) - set(train_idxs))
            self.train_seqs.append([seqs[i] for i in train_idxs])
            self.test_seqs.append([seqs[i] for i in test_idxs])

        for idx, sub_loc in enumerate(self.train_seqs):
            sublabel = np.zeros(len(self.train_seqs))
            sublabel[idx] = 1

            for seq in sub_loc:
                if flatten:
                    self.train_tensor.append(
                        (sum([seq2onehot(seq, self.aa_dict)], []), sublabel))
                else:
                    self.train_tensor.append(
                        (sum(seq2onehot(seq, self.aa_dict), []), sublabel))

        for idx, sub_loc in enumerate(self.test_seqs):
            sublabel = np.zeros(len(self.test_seqs))
            sublabel[idx] = 1
            sub_list = []

            for seq in sub_loc:
                if flatten:
                    sub_list.append(
                        [sum([seq2onehot(seq, self.aa_dict)], []), sublabel])
                else:
                    sub_list.append(
                        [sum(seq2onehot(seq, self.aa_dict), []), sublabel])

            self.test_tensor.append(sub_list)

        self.test_dict = {}
        for idx, label in enumerate(self.labels):
            self.test_dict[label] = self.test_tensor[idx]

        self.test_tensor = [seq for subloc in self.test_tensor for seq in
                            subloc]

    def print_labels(self):
        for label in self.labels:
            print (label)

