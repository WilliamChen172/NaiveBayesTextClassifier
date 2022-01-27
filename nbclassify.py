import math
import sys
import ast
import os
import glob

import nblearn

model_path = "nbmodel.txt"


def read_input(root):
    reviews = []
    paths = []

    pt_dir = os.path.join(root, "**/*.txt")
    for review in glob.glob(pt_dir, recursive=True):
        if review[-10:] == "README.txt":
            continue
        with open(review) as f:
            reviews.append(f.read())
            paths.append(review)
    return reviews, paths


def read_model():
    with open(model_path) as f:
        model = f.read()
    model = ast.literal_eval(model)
    return model


def predict(model, reviews, paths):
    results = []
    class_priors = model["class_priors"]
    vocabulary = model["vocabulary"]
    word_count = model["word_count"]
    token_count = model["token_count"]

    for review, path in zip(reviews, paths):
        tokens = nblearn.tokenize(review)
        pt_prob = math.log(class_priors["pt"])
        pd_prob = math.log(class_priors["pd"])
        nt_prob = math.log(class_priors["nt"])
        nd_prob = math.log(class_priors["nd"])
        for token in tokens:
            if token not in vocabulary:
                continue
            # Add Laplace smoothing
            pt = math.log((word_count["pt"].get(token, 0) + 1) / (token_count["pt"] + len(vocabulary)))
            pd = math.log((word_count["pd"].get(token, 0) + 1) / (token_count["pd"] + len(vocabulary)))
            nt = math.log((word_count["nt"].get(token, 0) + 1) / (token_count["nt"] + len(vocabulary)))
            nd = math.log((word_count["nd"].get(token, 0) + 1) / (token_count["nd"] + len(vocabulary)))
            pt_prob += pt
            pd_prob += pd
            nt_prob += nt
            nd_prob += nd

        # print(pt_prob, pd_prob, nt_prob, nd_prob)
        # print(pt_prob, nt_prob)
        # print(path)
        # if pt_prob > nt_prob:
        #     results.append("positive")
        # elif nt_prob > pt_prob:
        #     results.append("negative")
        if pt_prob == max(pt_prob, pd_prob, nt_prob, nd_prob):
            results.append("truthful positive")
        elif pd_prob == max(pt_prob, pd_prob, nt_prob, nd_prob):
            results.append("deceptive positive")
        elif nt_prob == max(pt_prob, pd_prob, nt_prob, nd_prob):
            results.append("truthful negative")
        elif nd_prob == max(pt_prob, pd_prob, nt_prob, nd_prob):
            results.append("deceptive negative")

    return results


def save_results(results, paths):
    with open("nboutput.txt", "w") as f:
        for result, path in zip(results, paths):
            f.write(result + " " + path + "\n")
        f.close()


if __name__ == '__main__':
    data_path = sys.argv[1]
    model = read_model()
    data, paths = read_input(data_path)
    results = predict(model, data, paths)
    save_results(results, paths)
