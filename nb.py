import sys
from collections import defaultdict, Counter
from math import log

class Classifier:
    def __init__(self):
        pass

    def test(self, samples):
        rights = 0
        wrongs = 0

        for real, features in samples:

            output = self.classify_one(features)
            pred = [el[1] for el in output]

            if real[0] == pred[0]:
                if VERBOSE:
                    print "correct TOP: real - "+str(real)+\
                        ", pred - "+str(pred)
                rights = rights + 1
            elif real[0] in pred:
                if VERBOSE:
                    print "correct GSS: real - "+str(real)+\
                        ", pred - "+str(pred)
                rights = rights + 1
            elif pred[0] in real:
                if VERBOSE: 
                    print "correct ALL: real - "+str(real)+\
                        ", pred - "+str(pred)
                rights = rights + 1
            else:
                if VERBOSE:
                    print "WRONG: real - "+str(real)+", pred - "+str(pred)
                wrongs = wrongs + 1

        if VERBOSE:
            print ""
            print "finished"
        print "correct: " + str(rights)
        print "wrong: " + str(wrongs)


#
# Implements a simple Multinomial Naive Bayes classifier
#
class NB(Classifier):
    def __init__(self):
        self.classes = Counter()
        self.features = defaultdict(Counter)
        self.f_totals = defaultdict(int)
        self.vocabulary = set()
        self.test_samples = 0

    def train_one(self, cl, ftrs):
        self.classes[cl] += 1
        self.features[cl].update(ftrs)
        self.vocabulary.update(ftrs.keys())
        self.test_samples += 1
        self.f_totals[cl] += sum(ftrs.values())
    
    def train(self, examples):
        for classes, features in examples:
            for cl in classes:
                self.train_one(cl,features)

    def classify_one(self, sample):
        probs = []
        for cl in self.classes:
            pr = log(float(self.classes[cl])/self.test_samples)
            for ftr in sample:
                pr = pr + log(float(self.features[cl][ftr]+1)/\
                    (self.f_totals[cl]+len(self.vocabulary)))
            probs.append((pr, cl))
        return sorted(probs)[::-1][:3]

    def classify(self, samples):
        ret = []
        for sample in samples:
            print sample
            ret.append(classify_one(sample))
        return ret


#
# Preprocesses the input data, removing the lowest level of classification, 
# and returning tuples of a list of classes and the feature dictionary.
#
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


def classtest(samples):
    t = 0
    g = 0
    m = 0
    w = 0

    for real, features in samples:
        cls = [x[0] for x in real]
        pr = classbayes.classify_one(features)
        pr = [x[1] for x in pr]

        if cls[0] == pr[0]:
            t += 1
        elif pr[0] in cls:
            g += 1
        elif cls[0] in pr:
            m += 1
        else:
            w += 1

    print t, g, m, w
    

class TwolevelNB(Classifier):
    def __init__(self):
        self.classbayes = NB()
        self.subclass = {\
            "a": NB(),\
            "b": NB(),\
            "c": NB(),\
            "d": NB(),\
            "e": NB(),\
            "f": NB(),\
            "g": NB(),\
            "h": NB()}

    def train(self, examples):
        for classes, features in examples:
            cls = set()
            for cl in classes:
                if cl[0] not in cls:
                    self.classbayes.train_one(cl[0], features)
                    cls.add(cl[0])
                self.subclass[cl[0]].train_one(cl, features)

    def classify_one(self, sample):
        classpred = self.classbayes.classify_one(sample)
        output = []
        for pr, cl in classpred:
            sub = self.subclass[cl].classify_one(sample)
            for s_pr, s_cl in sub:
                output.append((s_pr+pr, s_cl))

        return sorted(output)[::-1][:3]


if __name__ == "__main__":
    LEVELS = "1"
    TRAIN = "wipoalpha-train.txt"
    TEST = "wipoalpha-test.txt"
    VERBOSE = False

    if len(sys.argv) > 1:
        LEVELS = sys.argv[1]

    if len(sys.argv) > 3:
        TRAIN = sys.argv[2]
        TEST = sys.argv[3]

    if len(sys.argv) == 5 and sys.argv[4] == "verbose":
        VERBOSE = True
        
    classifier = NB() if LEVELS == "1" else TwolevelNB()

    with open(TRAIN, "r") as train_doc:
        train_samples = preprocess(train_doc.readlines())

    classifier.train(train_samples)

    with open(TEST, "r") as test_doc:
        test_samples = preprocess(test_doc.readlines())

    classifier.test(test_samples)

