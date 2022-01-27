import nblearn


model = "nbmodel.txt"

def read_model():
    with open(model)

def predict(self, reviews):
    result = []
    pt_prob = self.class_priors["pt"]
    pd_prob = self.class_priors["pd"]
    nt_prob = self.class_priors["nt"]
    nd_prob = self.class_priors["nd"]
    for review in reviews:
        tokens = self.tokenize(review)
        for token in tokens:
            if token not in self.vocabulary:
                continue

            # Add Laplace smoothing
            pt = (self.word_count["pt"].get(token, 0) + 1) / (self.review_count["pt"] + len(self.vocabulary))
            pd = (self.word_count["pd"].get(token, 0) + 1) / (self.review_count["pd"] + len(self.vocabulary))
            nt = (self.word_count["nt"].get(token, 0) + 1) / (self.review_count["nt"] + len(self.vocabulary))
            nd = (self.word_count["nd"].get(token, 0) + 1) / (self.review_count["nd"] + len(self.vocabulary))

            pt_prob *= pt
            pd_prob *= pd
            nt_prob *= nt
            nd_prob *= nd

    return

if __name__ == '__main__':
