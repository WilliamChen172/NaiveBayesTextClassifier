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


def read_input(root):
    reviews = []
    labels = []

    pt_dir = os.path.join(root, p_dir, tp_dir, "**/*.txt")
    for review in glob.glob(pt_dir):
        with open(review) as f:
            reviews.append(f.read())
            labels.append("pt")

    pd_dir = os.path.join(root, p_dir, d_dir, "**/*.txt")
    for review in glob.glob(pd_dir):
        with open(review) as f:
            reviews.append(f.read())
            labels.append("pd")

    nt_dir = os.path.join(root, n_dir, tn_dir, "**/*.txt")
    for review in glob.glob(nt_dir):
        with open(review) as f:
            reviews.append(f.read())
            labels.append("nt")

    nd_dir = os.path.join(root, n_dir, d_dir, "**/*.txt")
    for review in glob.glob(nd_dir):
        with open(review) as f:
            reviews.append(f.read())
            labels.append("nd")

    return reviews, labels


def tokenize(review):
    review = re.sub('[^a-z\s]+', ' ', review, flags=re.IGNORECASE)
    review = re.sub('(\s+)', ' ', review)
    review = review.lower()
    review = review.rstrip()
    tokens = re.split("\W+", review)
    return tokens


def fit(reviews, labels):
    nbmodel = {}
    review_count = {}
    class_priors = {}
    vocabulary = {}
    token_count = {"pt": 0, "pd": 0, "nt": 0, "nd": 0}
    word_count = {"pt": {}, "pd": {}, "nt": {}, "nd": {}}
    review_count["pt"] = sum(1 for label in labels if label == "pt")
    review_count["pd"] = sum(1 for label in labels if label == "pd")
    review_count["nt"] = sum(1 for label in labels if label == "nt")
    review_count["nd"] = sum(1 for label in labels if label == "nd")
    total_count = len(reviews)
    class_priors["pt"] = review_count["pt"] / total_count
    class_priors["pd"] = review_count["pd"] / total_count
    class_priors["nt"] = review_count["nt"] / total_count
    class_priors["nd"] = review_count["nd"] / total_count

    for review, label in zip(reviews, labels):
        tokens = tokenize(review)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = 0
            if token not in word_count[label]:
                word_count[label][token] = 0
            vocabulary[token] += 1
            word_count[label][token] += 1
            token_count[label] += 1

    nbmodel["token_count"] = token_count
    nbmodel["class_priors"] = class_priors
    nbmodel["vocabulary"] = vocabulary
    nbmodel["word_count"] = word_count
    with open('nbmodel.txt', 'w') as f:
        print(nbmodel, file=f)

    return


if __name__ == '__main__':
    data_path = sys.argv[1]
    data, labels = read_input(data_path)
    fit(data, labels)
