
from collections import defaultdict, Counter
from math import log

TRAIN = "wipoalpha-train.txt"
TEST = "wipoalpha-test.txt"


class NB:
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
                pr = pr + log(float(sample[ftr])*\
                    float(self.features[cl][ftr]+1)/\
                    (self.f_totals[cl]+len(self.vocabulary)))
            probs.append((pr, cl))
        return sorted(probs)[::-1][:3]

    def classify(self, samples):
        ret = []
        for sample in samples:
            print sample
            ret.append(classify_one(sample))
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
    

def test(samples):
    rights = 0
    wrongs = 0

    for real, features in samples:

        classpred = classbayes.classify_one(features)
        output = []
        for pr, cl in classpred:
            sub = subclass[cl].classify_one(features)
            for s_pr, s_cl in sub:
                output.append((s_pr+pr, s_cl))
            #output.append(sub[0])

        output = sorted(output)[::-1][:3]
        pred = [el[1] for el in output]


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
    classbayes = NB()
    subclass = {\
        "a": NB(),\
        "b": NB(),\
        "c": NB(),\
        "d": NB(),\
        "e": NB(),\
        "f": NB(),\
        "g": NB(),\
        "h": NB()}

    with open(TRAIN, "r") as train_doc:
        train_samples = preprocess(train_doc.readlines())

    for classes, features in train_samples:
        cls = set()
        for cl in classes:
            if cl[0] not in cls:
                classbayes.train_one(cl[0], features)
                cls.add(cl[0])
            subclass[cl[0]].train_one(cl, features)

    with open(TEST, "r") as test_doc:
        test_samples = preprocess(test_doc.readlines())

    test(test_samples)
    #classtest(test_samples)

