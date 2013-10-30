import collections
import logging
import itertools
import random


EvaluationConfig = collections.namedtuple('EvaluationConfig', ['num_samples', 'sample_size'])


def _sample(compound_list, size, seed):
        """
        Create a specific size sample from the compound list using a specific seed
        """
        return random.Random(seed).sample(compound_list, size)

class MorfessorEvaluationResult(object):
    def __init__(self, meta_data):
        self.meta_data = meta_data

        self.precision = []
        self.recall = []
        self.f_score = []
        self.sample_size = []

    def add_data_point(self, precision, recall, f_score, sample_size):
        self.precision.append(precision)
        self.recall.append(recall)
        self.f_score.append(f_score)
        self.sample_size.append(sample_size)

    # (precision | recall | fscore | sample_size ) (values | avg | min | max | std)
    # give option to format output
    # templates

class MorfessorEvaluation(object):

    def __init__(self, test_set):
        self.reference = {}

        for compound, analyses in test_set.items():
            self.reference[compound] = list(tuple(self._segmentation_indices(a)) for a in analyses)

        self._samples = {}

    def _create_samples(self, configuration=EvaluationConfig(10, 1000)):
        """
        Create, in a stable manner, n testsets of size x as defined in test_configuration
        """

        #TODO: test for the size of the training set. If too small, warn about it!

        compound_list = sorted(self.reference.keys())
        self.samples[configuration] = [_sample(compound_list, configuration.sample_size, i) for i in range(configuration.num_samples)]

    def get_samples(self, configuration=EvaluationConfig(10, 1000)):
        if not configuration in self._samples:
            self._create_samples(configuration)
        return self._samples[configuration]

    def _evaluate(self, prediction):
        wordlist = sorted(set(prediction.keys()) & set(self.reference.keys()))

        recall_sum = 0.0
        precis_sum = 0.0

        for word in wordlist:
            if len(word) < 2:
                continue

            recall_sum += max(((len(r) - len(set(r) - set(p))) / float(len(r)) if len(r) > 0 else 1.0)
                              for p, r in itertools.product(prediction[word],
                                                            self.reference[word])
                              )

            precis_sum += max(((len(p) - len(set(p) - set(r))) / float(len(set(p))) if len(p) > 0 else 1.0)
                              for p, r in itertools.product(prediction[word],
                                                            self.reference[word])
                              )

        precision = precis_sum / len(wordlist)
        recall = recall_sum / len(wordlist)
        f_score = 2.0/(1.0/precision+1.0/recall)

        return precision, recall, f_score, len(wordlist)

    @staticmethod
    def _segmentation_indices(annotation):
        cur_len = 0
        for a in annotation[:-1]:
            cur_len += len(a)
            yield cur_len

    @staticmethod
    def _merge_construction(construction):
        compound = construction[0]
        for i in range(1, len(construction)):
            compound = compound + construction[i]
        return compound

    def evaluate_model(self, model, configuration=EvaluationConfig(10, 1000), meta_data=None):
        mer = MorfessorEvaluationResult(meta_data)

        for sample in self.get_samples(configuration):
            prediction = {}
            for compound in sample:
                prediction[compound] = [tuple(self._segmentation_indices(model.viterbi_segment(compound)[0]))]

            mer.add_data_point(*self._evaluate(prediction))

        return mer

    def evaluate_segmentation(self, segmentation, meta_data=None):
        prediction = {}
        for construction in segmentation:
            prediction[self._merge_construction(construction)] = [tuple(self._segmentation_indices(construction))]

        return self._evaluate(prediction)






class WilcoxSignedRank(object):
    # params:
    # alpha
    pass

