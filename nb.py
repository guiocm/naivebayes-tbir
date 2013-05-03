import sys
from collections import defaultdict, Counter
from math import log

#
# Main Classifier class, implements common test functions.
#
class Classifier:
    def __init__(self):
        pass

    #
    # Evaluates the classification results
    # 
    def eval(self, real_cl, pred_cl, verbose):
	top = 0
	guess = 0
	ll = 0
        wrongs = 0
	rights = 0

        for real, pred in zip(real_cl, pred_cl):
	    corr = False

            if real[0] == pred[0]:
                if verbose:
                    print "correct TOP: real - " + str(real) + ", pred - " + str(pred)
                rights = rights + 1
		top += 1
		corr = True
            if real[0] in pred:
                if verbose:
                    print "correct GSS: real - " + str(real) + ", pred - " + str(pred)
                if not corr:
		    rights = rights + 1
		    corr = True
		guess += 1
            if pred[0] in real:
                if verbose: 
                    print "correct ALL: real - " + str(real) + ", pred - " + str(pred)
		if not corr:
		    rights = rights + 1
		    corr = True
		ll += 1
            if not corr:
                if verbose:
                    print "WRONG: real - " + str(real) + ", pred - " + str(pred)
                wrongs = wrongs + 1

        if verbose:
            print ""
            print "finished"
	
	total = rights+wrongs
        print "correct TOP: " + str(top) + " - " + str(float(top)/total)[:6]
        print "correct GSS: " + str(guess) + " - " + str(float(guess)/total)[:6]
        print "correct ALL: " + str(ll) + " - " + str(float(ll)/total)[:6]
        print "total correct: " + str(rights) + " - " + str(float(rights)/total)[:6]
        print "wrong: " + str(wrongs) + " - " + str(float(wrongs)/total)[:6]


    def strip_probs(self, classif):
	return [cl[1] for cl in classif]

    #
    # Classifies all samples (with implementation-dependant function)
    #
    def classify(self, samples):
	return [self.strip_probs(self.classify_one(sample)) for sample in samples]

    #
    # Main execution function, performs classification and output results to a file
    #
    def run(self, samples, verbose, result):
	real = [spl[0] for spl in samples]
	features = [spl[1] for spl in samples]

	pred = self.classify(features)

	with open(result, "w") as ofile:
	    for sample in pred:
		ofile.write(reduce(lambda x,y: x+" "+y, [el.upper() for el in sample])+"\n")
	    
	self.eval(real, pred, verbose)


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


#
# Implements a Classifier with a two-level hierarchy as present in the 
# patent classes.
#
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


#
# Top-level routine, reads and preprocesses the training and test files,
# performs the initialization and training of the classifier, and
# classifies and evaluates the performance on test samples.
#
if __name__ == "__main__":
    LEVELS = "1"
    TRAIN = "wipoalpha-train.txt"
    TEST = "wipoalpha-test.txt"
    RESULT = "result.txt"
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

    classifier.run(test_samples, VERBOSE, RESULT)


