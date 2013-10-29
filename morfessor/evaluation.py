import collections
import logging
import random

EvaluationConfig = collections.namedtuple('EvaluationConfig', ['num_samples', 'sample_size'])

class MorfessorEvaluation(object):

    def __init__(self, test_set, test_configuration=EvaluationConfig(10, 1000)):
        self.test_configuration = test_configuration

        self.reference = {}
        self.samples = []

        for compound, analyses in test_set:
            self.reference[compound] = analyses

        self._create_samples(sorted(self.reference.keys()))

    def _create_samples(self, compound_list):
        """
        Create, in a stable manner, n testsets of size x as defined in test_configuration
        """

        #TODO: test for the size of the training set. If too small, warn about it!

        for i in range(self.test_configuration.num_samples):
            self.samples.append(self._get_sample(compound_list, self.test_configuration.sample_size, i))

    @staticmethod
    def _get_sample(compound_list, size, seed):
        """
        Create, in a stable manner, n testsets of size x as defined in test_configuration
        """
        rand_class = random.Random(seed)
        return sorted(rand_class.sample(compound_list, size))

    @staticmethod
    def _segment_sample(sample, model):

        return []

    def evaluate(self, model):
        pass


class MorfessorEvaluationResult(object):
    pass




if __name__ == "__main__":
    me = MorfessorEvaluation(['a', 'b'])
    me._get_random_set(1,2)

