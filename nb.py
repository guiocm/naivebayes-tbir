
from collections import defaultdict, Counter
from math import log

TRAIN = "wipoalpha-train.txt"
TEST = "wipoalpha-test.txt"


class NB:
    def __init__(self):
        self.classes = Counter()
        self.features = defaultdict(Counter)
        self.f_totals = {}
        self.voc_size = 61499
    
    def train(self, examples):
        for classes, features in examples:
            for cl in classes:
                self.classes[cl] += 1
                self.features[cl].update(features)

        self.test_samples = sum(self.classes.values())

        for cl in self.features:
            self.f_totals[cl] = sum(self.features[cl].values())

    def classify(self, samples):
        ret = []
        for sample in samples:
            print sample
            probs = []
            for cl in self.classes:
                pr = log(float(self.classes[cl])/self.test_samples)
                for ftr in sample:
                    pr = pr + log(float(sample[ftr])*\
                        float(self.features[cl][ftr]+1)/\
                        (self.f_totals[cl]+self.voc_size))
                probs.append((pr, cl))
            ret.append(sorted(probs)[::-1][:3])

        return ret


def preprocess(data):
    samples = []
    for line in data:
        parts = line.split(" ")
        classes = parts.pop(0)
        classes = [c[:-1] for c in classes.split(",")]

        parts = [el.split(":") for el in parts]
        features = {int(k):int(v) for k,v in parts}

        samples.append((classes, features))

    return samples


def test(classifier, samples):
    classes = [el[0] for el in samples]
    features = [el[1] for el in samples]

    output = classifier.classify(features)
    predicted = [map(lambda x: x[1], el) for el in output]

    rights = 0
    wrongs = 0

    for real, pred in zip(classes, predicted):
        if real[0] == pred[0]:
            print "correct TOP: real - "+str(real)+", pred - "+str(pred)
            rights = rights + 1
        elif real[0] in pred:
            print "correct GSS: real - "+str(real)+", pred - "+str(pred)
            rights = rights + 1
        elif pred[0] in real:
            print "correct ALL: real - "+str(real)+", pred - "+str(pred)
            rights = rights + 1
        else:
            print "WRONG: real - "+str(real)+", pred - "+str(pred)
            wrongs = wrongs + 1

    print ""
    print "finished"
    print "correct: " + str(rights)
    print "wrong: " + str(wrongs)





if __name__ == "__main__":
    naivebayes = NB()

    with open(TRAIN, "r") as train_doc:
        train_samples = preprocess(train_doc.readlines())

    naivebayes.train(train_samples)

    with open(TEST, "r") as test_doc:
        test_samples = preprocess(test_doc.readlines())

    test(naivebayes, test_samples)


