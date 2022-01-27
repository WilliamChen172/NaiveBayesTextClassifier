import numpy as np
import sys
import os
import re
import glob

p_dir = "positive_polarity"
n_dir = "negative_polarity"
tp_dir = "truthful_from_TripAdvisor"
tn_dir = "truthful_from_Web"
d_dir = "deceptive_from_MTurk"
fold1 = "fold1"
fold2 = "fold2"
fold3 = "fold3"
fold4 = "fold4"
# labels: 0 - pt, 1 - pd, 2 - nt, 3 - nd


class NaiveBayes:

    def read_input(self, root):
        reviews = []
        labels = []

        pt_dir = os.path.join(root, p_dir, tp_dir, "fold[2-4]", "*.txt")
        for review in glob.glob(pt_dir):
            with open(review) as f:
                reviews.append(f.read())
                labels.append(0)

        pd_dir = os.path.join(root, p_dir, d_dir, "fold[2-4]", "*.txt")
        for review in glob.glob(pd_dir):
            with open(review) as f:
                reviews.append(f.read())
                labels.append(1)

        nt_dir = os.path.join(root, n_dir, tn_dir, "fold[2-4]", "*.txt")
        for review in glob.glob(nt_dir):
            with open(review) as f:
                reviews.append(f.read())
                labels.append(2)

        nd_dir = os.path.join(root, n_dir, d_dir, "fold[2-4]", "*.txt")
        for review in glob.glob(nd_dir):
            with open(review) as f:
                reviews.append(f.read())
                labels.append(3)

        return reviews, labels

    def preprocess(self, data):

        return



















if __name__ == '__main__':
    data_path = sys.argv[1]
    nbmodel = NaiveBayes()
    data, labels = nbmodel.read_input(data_path)
    print(len(data), len(labels))
