from __future__ import print_function

import collections
import logging
from itertools import chain, product
import math
import random

_logger = logging.getLogger(__name__)

EvaluationConfig = collections.namedtuple('EvaluationConfig',
                                          ['num_samples', 'sample_size'])

FORMAT_STRINGS = {
    'default': """Filename\t: {name}
Num samples\t: {samplesize_count}
Sample size\t: {samplesize_avg}
F-score\t\t: {fscore_avg:.3}
Precision\t: {precision_avg:.3}
Recall\t\t: {recall_avg:.3}""",
    'table': "{name:10} {fscore_avg:6.3} {precision_avg:6.3} {recall_avg:6.3}",
    'latex': "{name} & {fscore_avg:.3} & {precision_avg:.3} &"
             " {recall_avg:.3}\\\\"}


def _sample(compound_list, size, seed):
    """
    Create a specific size sample from the compound list using a specific seed
    """
    return random.Random(seed).sample(compound_list, size)


class MorfessorEvaluationResult(object):
    print_functions = {'avg': lambda x: sum(x) / len(x),
                       'min': min,
                       'max': max,
                       'values': list,
                       'count': len}

    def __init__(self, meta_data):
        self.meta_data = meta_data

        self.precision = []
        self.recall = []
        self.fscore = []
        self.samplesize = []

    def __getitem__(self, item):
        d = self._get_data_mat()
        d.update(self.meta_data)
        return d[item]

    def add_data_point(self, precision, recall, f_score, sample_size):
        self.precision.append(precision)
        self.recall.append(recall)
        self.fscore.append(f_score)
        self.samplesize.append(sample_size)

    def __str__(self):
        return self.format(FORMAT_STRINGS['default'])

    def _get_data_mat(self):
        return {'{}_{}'.format(value, func_name): func(getattr(self, value))
                for value in ('precision', 'recall', 'fscore', 'samplesize')
                for func_name, func in self.print_functions.items()}

    def format(self, format_string):
        d = self._get_data_mat()
        d.update(self.meta_data)
        return format_string.format(**d)


class MorfessorEvaluation(object):
    def __init__(self, test_set):
        self.reference = {}

        for compound, analyses in test_set.items():
            self.reference[compound] = list(
                tuple(self._segmentation_indices(a)) for a in analyses)

        self._samples = {}

    def _create_samples(self, configuration=EvaluationConfig(10, 1000)):
        """
        Create, in a stable manner, n testsets of size x as defined in
        test_configuration
        """

        #TODO: test for the size of the training set. If too small,
        # warn about it!

        compound_list = sorted(self.reference.keys())
        self._samples[configuration] = [
            _sample(compound_list, configuration.sample_size, i) for i in
            range(configuration.num_samples)]

    def get_samples(self, configuration=EvaluationConfig(10, 1000)):
        """
        Get a list of samples. A sample is a list of compounds.

        This method is stable, so each time it is called with a specific
        test_set and configuration it will return the same samples. Also this
        method caches the samples in the _samples variable.
        """
        if not configuration in self._samples:
            self._create_samples(configuration)
        return self._samples[configuration]

    def _evaluate(self, prediction):

        def calc_prop_distance(ref, pred):
            #TODO rename variables
            if len(ref) == 0:
                return 1.0
            diff = len(set(ref) - set(pred))
            return (len(ref) - diff) / float(len(ref))

        wordlist = sorted(set(prediction.keys()) & set(self.reference.keys()))

        recall_sum = 0.0
        precis_sum = 0.0

        for word in wordlist:
            if len(word) < 2:
                continue

            recall_sum += max(calc_prop_distance(r, p)
                              for p, r in product(prediction[word],
                                                  self.reference[word]))

            precis_sum += max(calc_prop_distance(p, r)
                              for p, r in product(prediction[word],
                                                  self.reference[word]))

        precision = precis_sum / len(wordlist)
        recall = recall_sum / len(wordlist)
        f_score = 2.0 / (1.0 / precision + 1.0 / recall)

        return precision, recall, f_score, len(wordlist)

    @staticmethod
    def _segmentation_indices(annotation):
        cur_len = 0
        for a in annotation[:-1]:
            cur_len += len(a)
            yield cur_len

    def evaluate_model(self, model, configuration=EvaluationConfig(10, 1000),
                       meta_data=None):
        mer = MorfessorEvaluationResult(meta_data)

        for i, sample in enumerate(self.get_samples(configuration)):
            _logger.debug("Evaluating sample {}".format(i))
            prediction = {}
            for compound in sample:
                prediction[compound] = [tuple(self._segmentation_indices(
                    model.viterbi_segment(compound)[0]))]

            mer.add_data_point(*self._evaluate(prediction))

        return mer

    def evaluate_segmentation(self, segmentation, meta_data=None):
        def merge_constructions(constructions):
            compound = constructions[0]
            for i in range(1, len(constructions)):
                compound = compound + constructions[i]
            return compound

        prediction = {}
        for analysis in segmentation:
            prediction[merge_constructions(analysis)] = [
                tuple(self._segmentation_indices(analysis))]

        return self._evaluate(prediction)


class WilcoxSignedRank(object):
    # params:
    # alpha
    def __init__(self, alpha=0.01):
        pass

    def _wilcox(self, d):
        count = len(d)
        ranks = self._rankdata([abs(v) for v in d])
        rank_sum_pos = sum(r for r, v in zip(ranks, d) if v > 0)
        rank_sum_neg = sum(r for r, v in zip(ranks, d) if v < 0)

        test = min(rank_sum_neg, rank_sum_pos)

        mean = count * (count + 1) * 0.25
        stdev = math.sqrt((count*(count + 1) * (2 * count + 1)) / 24.0)

        correction = +0.5 if test > mean else -0.5
        z = (test - mean - correction) / stdev

        return 2 * self.norm_cum_pdf(abs(z))

    def _rankdata(self, d):
        od = collections.Counter()
        for v in d:
            od[v] += 1

        rank_dict = {}
        cur_rank = 1
        for val, count in sorted(od.items(), key=lambda x: x[0]):
            rank_dict[val] = (cur_rank + (cur_rank + count - 1)) / 2
            cur_rank += count

        return [rank_dict[v] for v in d]

    def norm_cum_pdf(self, z):
        return 0.5 - 0.5 * math.erf(z / math.sqrt(2))

    def significance_test(self, evaluations, val_property='fscore_values',
                          name_property='name'):

        results = {r[name_property]: r[val_property] for r in evaluations}
        p = {}
        for r1, r2 in product(results.keys(), results.keys()):
            p[(r1, r2)] = self._wilcox([v1-v2 for v1, v2 in zip(results[r1],
                                                                results[r2])])

        return p

    @staticmethod
    def print_table(results):
        names = sorted(set(r[0] for r in results.keys()))

        col_width = max(max(len(n) for n in names), 5)

        for h in chain([""], names):
            print('{:{width}}'.format(h, width=col_width), end='|')
        print()

        for name in names:
            print('{:{width}}'.format(name, width=col_width), end='|')

            for name2 in names:
                print('{:{width}.5}'.format(results[(name, name2)],
                                            width=col_width), end='|')
            print()
