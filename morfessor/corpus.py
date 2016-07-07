"""Implementations for corpus and lexicon encoding and weighting"""

import collections
import logging
import math
import re

from .utils import segmentation_to_splitloc, _progress

_logger = logging.getLogger(__name__)


class Encoding(object):
    """Base class for calculating the entropy (encoding length) of a corpus
    or lexicon.

    Commonly subclassed to redefine specific methods.

    """
    def __init__(self, weight=1.0):
        """Initizalize class

        Arguments:
            weight: weight used for this encoding
        """
        self.logtokensum = 0.0
        self.tokens = 0
        self.boundaries = 0
        self.weight = weight

    # constant used for speeding up logfactorial calculations with Stirling's
    # approximation
    _log2pi = math.log(2 * math.pi)

    @property
    def types(self):
        """Define number of types as 0. types is made a property method to
        ensure easy redefinition in subclasses

        """
        return 0

    @classmethod
    def _logfactorial(cls, n):
        """Calculate logarithm of n!.

        For large n (n > 20), use Stirling's approximation.

        """
        if n < 2:
            return 0.0
        if n < 20:
            return math.log(math.factorial(n))
        logn = math.log(n)
        return n * logn - n + 0.5 * (logn + cls._log2pi)

    def frequency_distribution_cost(self):
        """Calculate -log[(u - 1)! (v - u)! / (v - 1)!]

        v is the number of tokens+boundaries and u the number of types

        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens + self.boundaries
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 1) -
                self._logfactorial(tokens - self.types))

    def permutations_cost(self):
        """The permutations cost for the encoding."""
        return -self._logfactorial(self.boundaries)

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the encoding."""
        self.tokens += new_count - old_count
        if old_count > 1:
            self.logtokensum -= old_count * math.log(old_count)
        if new_count > 1:
            self.logtokensum += new_count * math.log(new_count)

    def get_cost(self):
        """Calculate the cost for encoding the corpus/lexicon"""
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum
                 + self.permutations_cost()) * self.weight
                + self.frequency_distribution_cost())


class CorpusEncoding(Encoding):
    """Encoding the corpus class

    The basic difference to a normal encoding is that the number of types is
    not stored directly but fetched from the lexicon encoding. Also does the
    cost function not contain any permutation cost.
    """
    def __init__(self, lexicon_encoding, weight=1.0):
        super(CorpusEncoding, self).__init__(weight)
        self.lexicon_encoding = lexicon_encoding

    @property
    def types(self):
        """Return the number of types of the corpus, which is the same as the
         number of boundaries in the lexicon + 1

        """
        return self.lexicon_encoding.boundaries + 1

    def frequency_distribution_cost(self):
        """Calculate -log[(M - 1)! (N - M)! / (N - 1)!] for M types and N
        tokens.

        """
        if self.types < 2:
            return 0.0
        tokens = self.tokens
        return (self._logfactorial(tokens - 1) -
                self._logfactorial(self.types - 2) -
                self._logfactorial(tokens - self.types + 1))

    def get_cost(self):
        """Override for the Encoding get_cost function. A corpus does not
        have a permutation cost

        """
        if self.boundaries == 0:
            return 0.0

        n = self.tokens + self.boundaries
        return ((n * math.log(n)
                 - self.boundaries * math.log(self.boundaries)
                 - self.logtokensum) * self.weight
                + self.frequency_distribution_cost())


class AnnotatedCorpusEncoding(Encoding):
    """Encoding the cost of an Annotated Corpus.

    In this encoding constructions that are missing are penalized.

    """
    def __init__(self, corpus_coding, weight=None, penalty=-9999.9):
        """
        Initialize encoding with appropriate meta data

        Arguments:
            corpus_coding: CorpusEncoding instance used for retrieving the
                             number of tokens and boundaries in the corpus
            weight: The weight of this encoding. If the weight is None,
                      it is updated automatically to be in balance with the
                      corpus
            penalty: log penalty used for missing constructions

        """
        super(AnnotatedCorpusEncoding, self).__init__()
        self.do_update_weight = True
        self.weight = 1.0
        if weight is not None:
            self.do_update_weight = False
            self.weight = weight
        self.corpus_coding = corpus_coding
        self.penalty = penalty
        self.constructions = collections.Counter()

    def set_constructions(self, constructions):
        """Method for re-initializing the constructions. The count of the
        constructions must still be set with a call to set_count

        """
        self.constructions = constructions
        self.tokens = sum(constructions.values())
        self.logtokensum = 0.0

    def set_count(self, construction, count):
        """Set an initial count for each construction. Missing constructions
        are penalized
        """
        annot_count = self.constructions[construction]
        if count > 0:
            self.logtokensum += annot_count * math.log(count)
        else:
            self.logtokensum += annot_count * self.penalty

    def update_count(self, construction, old_count, new_count):
        """Update the counts in the Encoding, setting (or removing) a penalty
         for missing constructions

        """
        if construction in self.constructions:
            annot_count = self.constructions[construction]
            if old_count > 0:
                self.logtokensum -= annot_count * math.log(old_count)
            else:
                self.logtokensum -= annot_count * self.penalty
            if new_count > 0:
                self.logtokensum += annot_count * math.log(new_count)
            else:
                self.logtokensum += annot_count * self.penalty

    def update_weight(self):
        """Update the weight of the Encoding by taking the ratio of the
        corpus boundaries and annotated boundaries
        """
        if not self.do_update_weight:
            return
        old = self.weight
        self.weight = (self.corpus_coding.weight *
                       float(self.corpus_coding.boundaries) / self.boundaries)
        if self.weight != old:
            _logger.info("Corpus weight of annotated data set to %s"
                         % self.weight)

    def get_cost(self):
        """Return the cost of the Annotation Corpus."""
        if self.boundaries == 0:
            return 0.0
        n = self.tokens + self.boundaries
        return ((n * math.log(self.corpus_coding.tokens +
                              self.corpus_coding.boundaries)
                 - self.boundaries * math.log(self.corpus_coding.boundaries)
                 - self.logtokensum) * self.weight)


class LexiconEncoding(Encoding):
    """Class for calculating the encoding cost for the Lexicon"""

    def __init__(self):
        """Initialize Lexcion Encoding"""
        super(LexiconEncoding, self).__init__()
        self.atoms = collections.Counter()

    @property
    def types(self):
        """Return the number of different atoms in the lexicon + 1 for the
        compound-end-token

        """
        return len(self.atoms) + 1

    def add(self, construction):
        """Add a construction to the lexicon, updating automatically the
        count for its atoms

        """
        self.boundaries += 1
        for atom in construction:
            c = self.atoms[atom]
            self.atoms[atom] = c + 1
            self.update_count(atom, c, c + 1)

    def remove(self, construction):
        """Remove construction from the lexicon, updating automatically the
        count for its atoms

        """
        self.boundaries -= 1
        for atom in construction:
            c = self.atoms[atom]
            self.atoms[atom] = c - 1
            self.update_count(atom, c, c - 1)

    def get_codelength(self, construction):
        """Return an approximate codelength for new construction."""
        l = len(construction) + 1
        cost = l * math.log(self.tokens + l)
        cost -= math.log(self.boundaries + 1)
        for atom in construction:
            if atom in self.atoms:
                c = max(1, self.atoms[atom])
            else:
                c = 1
            cost -= math.log(c)
        return cost


class CorpusWeight(object):
    @classmethod
    def move_direction(cls, model, direction, epoch):
        if direction != 0:
            weight = model.get_corpus_coding_weight()
            if direction > 0:
                weight *= 1 + 2.0 / epoch
            else:
                weight *= 1.0 / (1 + 2.0 / epoch)
            model.set_corpus_coding_weight(weight)
            _logger.info("Corpus weight set to {}".format(weight))
            return True
        return False


class FixedCorpusWeight(CorpusWeight):
    def __init__(self, weight):
        self.weight = weight

    def update(self, model, _):
        model.set_corpus_coding_weight(self.weight)
        return False


class AlignedTokenCountCorpusWeight(CorpusWeight):
    """Class for using a sentence-aligned parallel bilingual corpus
    to set the corpus weight in such a way that the number of
    morphs in corpus of the language to be segmented
    is as similar as possible to the number of tokens on the reference side.
    """
    re_token_sep = re.compile(r'\s+', re.UNICODE)
    align_losses = ('abs', 'square', 'zeroone', 'tot')

    def __init__(self,
                 unsegmented_dev,
                 reference_dev,
                 threshold=0.01,
                 loss='abs',
                 linguistic_dev=None):
        self.unsegmented_dev = list(self.tokenize(unsegmented_dev))
        self.reference_counts = list(len(x) for x
                                     in self.tokenize(reference_dev))
        _logger.info('Total reference tokens {}'.format(
            sum(self.reference_counts)))
        self.threshold = threshold
        self.align_loss_idx = self.align_losses.index(loss)
        assert len(self.unsegmented_dev) == len(self.reference_counts)
        self.previous_weight = None
        self.previous_cost = None
        self.previous_d = None
        if linguistic_dev is not None:
            self.linguistic_dev = list(self.tokenize(linguistic_dev))
            assert len(self.linguistic_dev) == len(self.reference_counts)
        else:
            self.linguistic_dev = None


    def update(self, model, epoch):
        if epoch < 1:
            # Can't use viterbi_segment before first epoch
            return False
        weight = model.get_corpus_coding_weight()
        (cost, d) = self.evaluation(model)
        if self.previous_cost is not None:
            absdiff = abs(cost - self.previous_cost)
            absthresh = self.previous_cost * self.threshold
            if absdiff < absthresh:
                _logger.info("Align cost delta {} is below threshold {}. "
                    "Weight learning stopped".format(absdiff, absthresh))
                return False
        if self.previous_weight is None or cost < self.previous_cost:
            # accept the previous step
            self.previous_weight = weight
            self.previous_cost = cost
            self.previous_d = d
            _logger.info("Accepting step to {}".format(weight))
        else:
            # revert the previous step
            weight = self.previous_weight
            _logger.info("Reverting weight to {}".format(weight))
            model.set_corpus_coding_weight(weight)
            cost = self.previous_cost
            d = self.previous_d
        # new step
        return self.move_direction(model, d, epoch)

    @classmethod
    def tokenize(cls, lines):
        for line in lines:
            line = line.strip()
            yield cls.re_token_sep.split(line)

    def evaluation(self, model):
        costs, d, _ = self.calculate_costs(model)
        cost = costs[self.align_loss_idx]
        return (cost, d)

    def calculate_costs(self, model):
        abs_cost = 0.0
        sq_cost = 0.0
        zeroone_cost = 0.0
        tot_cost = 0.0
        direction = 0
        tot_tokens = 0
        cache = {}
        if self.linguistic_dev is not None:
            self.morph_totals = collections.Counter()
            self.morph_scores_pos = collections.Counter()
            self.morph_scores_neg = collections.Counter()
            linguistic_dev_iter = iter(self.linguistic_dev)
        _logger.info('Segmenting aligned parallel corpus for weight learning')
        for (tokens, ref) in zip(_progress(self.unsegmented_dev),
                                 self.reference_counts):
            segments = collections.Counter()
            for w in tokens:
                segments.update(self._cached_seg(model, cache, w))
            segcount = sum(segments.values())
            tot_tokens += segcount
            diff = segcount - ref
            if diff > 0:
                d = 1
            elif diff < 0:
                d = -1
            else:
                d = 0
            direction += d
            abs_cost += abs(diff)
            sq_cost += diff**2
            if diff != 0:
                zeroone_cost += 1
            tot_cost += diff
            if self.linguistic_dev is not None:
                # also count morph-type-level scores
                ling_morphs = collections.Counter(next(linguistic_dev_iter))
                self.morph_totals.update(ling_morphs)
                # Observe: - operator (as opposed to .subtract)
                #   uses multiset semantics, 
                #   and will not result in negative counts.
                not_in_seg = ling_morphs - segments
                in_seg = ling_morphs - not_in_seg
                if diff > 0:
                    # oversegmented
                    for morph in ling_morphs:
                        # strong plus if split in an overseg sentence
                        self.morph_scores_pos[morph] += in_seg[morph]
                        # weak plus if joined in an overseg sentence
                        self.morph_scores_neg[morph] -= not_in_seg[morph]
                elif diff < 0:
                    # undersegmented
                    for morph in ling_morphs:
                        # strong minus if joined in an underseg sentence
                        self.morph_scores_neg[morph] += not_in_seg[morph]
                        # weak minus if split in an underseg sentence
                        self.morph_scores_pos[morph] -= in_seg[morph]
        tot_cost = abs(tot_cost)
        costs = (abs_cost, sq_cost, zeroone_cost, tot_cost)
        _logger.info('Align costs {}, direction {}, total tokens {}'.format(
            costs, direction, tot_tokens))
        return (costs, direction, tot_tokens)

    def _cached_seg(self, model, cache, word):
        if word not in cache:
            try:
                seg = model.segment(word)
            except (KeyError, AttributeError):
                # don't use viterbi_segment: the only unseen words should be
                # unanalyzable words, which are not split anyhow
                #seg = model.viterbi_segment(word)[0]
                seg = [word]
            cache[word] = seg
        return cache[word]


class AnnotationCorpusWeight(CorpusWeight):
    """Class for using development annotations to update the corpus weight
    during batch training

    """

    def __init__(self, devel_set, threshold=0.01):
        self.data = devel_set
        self.threshold = threshold

    def update(self, model, epoch):
        """Tune model corpus weight based on the precision and
        recall of the development data, trying to keep them equal"""
        if epoch < 1:
            return False
        tmp = self.data.items()
        wlist, annotations = zip(*tmp)
        segments = [model.viterbi_segment(w)[0] for w in wlist]
        d = self._estimate_segmentation_dir(segments, annotations)

        return self.move_direction(model, d, epoch)

    @classmethod
    def _boundary_recall(cls, prediction, reference):
        """Calculate average boundary recall for given segmentations."""
        rec_total = 0
        rec_sum = 0.0
        for pre_list, ref_list in zip(prediction, reference):
            best = -1
            for ref in ref_list:
                # list of internal boundary positions
                ref_b = set(segmentation_to_splitloc(ref))
                if len(ref_b) == 0:
                    best = 1.0
                    break
                for pre in pre_list:
                    pre_b = set(segmentation_to_splitloc(pre))
                    r = len(ref_b.intersection(pre_b)) / float(len(ref_b))
                    if r > best:
                        best = r
            if best >= 0:
                rec_sum += best
                rec_total += 1
        return rec_sum, rec_total

    @classmethod
    def _bpr_evaluation(cls, prediction, reference):
        """Return boundary precision, recall, and F-score for segmentations."""
        rec_s, rec_t = cls._boundary_recall(prediction, reference)
        pre_s, pre_t = cls._boundary_recall(reference, prediction)
        rec = rec_s / rec_t
        pre = pre_s / pre_t
        f = 2.0 * pre * rec / (pre + rec)
        return pre, rec, f

    def _estimate_segmentation_dir(self, segments, annotations):
        """Estimate if the given compounds are under- or oversegmented.

        The decision is based on the difference between boundary precision
        and recall values for the given sample of segmented data.

        Arguments:
          segments: list of predicted segmentations
          annotations: list of reference segmentations

        Return 1 in the case of oversegmentation, -1 in the case of
        undersegmentation, and 0 if no changes are required.

        """
        pre, rec, f = self._bpr_evaluation([[x] for x in segments],
                                           annotations)
        _logger.info("Boundary evaluation: precision %.4f; recall %.4f" %
                     (pre, rec))
        if abs(pre - rec) < self.threshold:
            return 0
        elif rec > pre:
            return 1
        else:
            return -1


class MorphLengthCorpusWeight(CorpusWeight):
    def __init__(self, morph_lenght, threshold=0.01):
        self.morph_length = morph_lenght
        self.threshold = threshold

    def update(self, model, epoch):
        if epoch < 1:
            return False
        cur_length = self.calc_morph_length(model)

        _logger.info("Current morph-length: {}".format(cur_length))

        if (abs(self.morph_length - cur_length) / self.morph_length >
                self.threshold):
            d = abs(self.morph_length - cur_length) / (self.morph_length
                                                       - cur_length)
            return self.move_direction(model, d, epoch)
        return False

    @classmethod
    def calc_morph_length(cls, model):
        total_constructions = 0
        total_atoms = 0
        for compound in model.get_compounds():
            constructions = model.segment(compound)
            for construction in constructions:
                total_constructions += 1
                total_atoms += len(construction)
        if total_constructions > 0:
            return float(total_atoms) / total_constructions
        else:
            return 0.0


class NumMorphCorpusWeight(CorpusWeight):
    def __init__(self, num_morph_types, threshold=0.01):
        self.num_morph_types = num_morph_types
        self.threshold = threshold

    def update(self, model, epoch):
        if epoch < 1:
            return False
        cur_morph_types = model._lexicon_coding.boundaries

        _logger.info("Number of morph types: {}".format(cur_morph_types))


        if (abs(self.num_morph_types - cur_morph_types) / self.num_morph_types
                > self.threshold):
            d = (abs(self.num_morph_types - cur_morph_types) /
                 (self.num_morph_types - cur_morph_types))
            return self.move_direction(model, d, epoch)
        return False
