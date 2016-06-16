import collections
import heapq
import logging
import math
import numbers
import random
import re

from .corpus import LexiconEncoding, CorpusEncoding, \
    AnnotatedCorpusEncoding, FixedCorpusWeight
from .utils import segmentation_to_splitloc, splitloc_to_segmentation, \
    _progress, constructions_to_str
from .exception import MorfessorException, SegmentOnlyModelException

_logger = logging.getLogger(__name__)


# count = count of the node
# splitloc = integer or tuple. Location(s) of the possible splits for virtual
#            constructions; empty tuple or 0 if real construction
ConstrNode = collections.namedtuple('ConstrNode', ['count', 'splitloc'])


class BaselineModel(object):
    """Morfessor Baseline model class.

    Implements training of and segmenting with a Morfessor model. The model
    is complete agnostic to whether it is used with lists of strings (finding
    phrases in sentences) or strings of characters (finding morphs in words).

    """

    penalty = -9999.9

    def __init__(self, forcesplit_list=None, corpusweight=None,
                 use_skips=False, nosplit_re=None):
        """Initialize a new model instance.

        Arguments:
            forcesplit_list: force segmentations on the characters in
                               the given list
            corpusweight: weight for the corpus cost
            use_skips: randomly skip frequently occurring constructions
                         to speed up training
            nosplit_re: regular expression string for preventing splitting
                          in certain contexts

        """
        # Stores ConstrNode for each compound in training data.
        self._compounds = {}

        # In analyses for each construction a ConstrNode is
        # stored. Real constructions have no split locations.
        self._analyses = {}

        # Flag to indicate the model is only useful for segmentation
        self._segment_only = False

        # Cost variables
        self._lexicon_coding = LexiconEncoding()
        self._corpus_coding = CorpusEncoding(self._lexicon_coding)
        self._annot_coding = None

        #Set corpus weight updater
        self.set_corpus_weight_updater(corpusweight)

        # Configuration variables
        self._use_skips = use_skips  # Random skips for frequent constructions
        self._supervised = False
        self._restricted = False

        # Counter for random skipping
        self._counter = collections.Counter()
        if forcesplit_list is None:
            self.forcesplit_list = []
        else:
            self.forcesplit_list = forcesplit_list
        if nosplit_re is None:
            self.nosplit_re = None
        else:
            self.nosplit_re = re.compile(nosplit_re, re.UNICODE)

    def set_corpus_weight_updater(self, corpus_weight):
        if corpus_weight is None:
            self._corpus_weight_updater = FixedCorpusWeight(1.0)
        elif isinstance(corpus_weight, numbers.Number):
            self._corpus_weight_updater = FixedCorpusWeight(corpus_weight)
        else:
            self._corpus_weight_updater = corpus_weight

        self._corpus_weight_updater.update(self, 0)

    def _check_segment_only(self):
        if self._segment_only:
            raise SegmentOnlyModelException()

    def _check_integrity(self):
        failed = False
        parents = collections.defaultdict(list)
        # Check compounds
        for compound in self._compounds.keys():
            node = self._compounds[compound]
            if node.splitloc:
                children = splitloc_to_segmentation(compound, node.splitloc)
            else:
                children = [compound]
            for part in children:
                if not part in self._analyses:
                    _logger.critical(
                        "Construction '%s' of compound '%s' not found",
                        part, compound)
                    failed = True
                elif self._analyses[part].count < node.count:
                    _logger.critical(
                        ("Construction '%s' has lower "
                         "count than parent compound '%s': %s < %s"),
                        part, compound, self._analyses[part].count,
                        node.count)
                    failed = True
                parents[part].append((compound, node.count))
        # Check constructions
        for construction in self._analyses.keys():
            node = self._analyses[construction]
            if node.count < 1:
                _logger.critical("Non-positive count %s for construction %s",
                                 node.count, construction)
                failed = True
            if not node.splitloc:
                continue
            for part in splitloc_to_segmentation(construction, node.splitloc):
                if not part in self._analyses:
                    _logger.critical(
                        "Subconstruction '%s' of '%s' not found",
                        part, construction)
                    failed = True
                elif self._analyses[part].count < node.count:
                    _logger.critical(
                        ("Subconstruction '%s' has lower "
                         "count than parent '%s': %s < %s"),
                        part, construction, self._analyses[part].count,
                        node.count)
                    failed = True
                parents[part].append((construction, node.count))
        # Check count sums
        for construction in parents.keys():
            node = self._analyses[construction]
            psum = sum([x[1] for x in parents[construction]])
            if psum != node.count:
                _logger.critical(
                    ("Counts %s of construction '%s' does not "
                     "match counts of parents %s"),
                    node.count, construction, parents[construction]
                    if len(parents[construction]) < 20 else "")
                failed = True
        if failed:
            raise MorfessorException("Corrupted model")

    def _check_restrictions(self):
        if not self._restricted:
            return
        violations = 0
        rtotal = 0
        stotal = 0
        for _, compound, segmentation in self.get_segmentations():
            if compound in self.allowed_boundaries:
                rtotal += len(compound) - 1 - \
                          len(self.allowed_boundaries[compound])
                stotal += len(segmentation) - 1
                for idx in segmentation_to_splitloc(segmentation):
                    if idx not in self.allowed_boundaries[compound]:
                        violations += 1
        if rtotal > 0 and stotal > 0:
            _logger.info(
                ("Violated restrictions: %s "
                 "(%.2f%% of restrictions, %.2f%% of segmentations)"),
                violations, 100*violations/rtotal, 100*violations/stotal)

    @property
    def tokens(self):
        """Return the number of construction tokens."""
        return self._corpus_coding.tokens

    @property
    def types(self):
        """Return the number of construction types."""
        return self._corpus_coding.types - 1  # do not include boundary

    def _add_compound(self, compound, count):
        """Add compound with count to data."""
        if compound in self._compounds:
            oldc = self._compounds[compound].count
            self._compounds[compound] = \
                self._compounds[compound]._replace(count=oldc+count)
        else:
            self._compounds[compound] = ConstrNode(count, tuple())
        self._corpus_coding.boundaries += count
        if self._compounds[compound].splitloc:
            constructions = splitloc_to_segmentation(
                compound, self._compounds[compound].splitloc)
        else:
            constructions = [compound]
        for construction in constructions:
            self._modify_construction_count(construction, count)

    def _remove(self, construction):
        """Remove construction from model."""
        count, splitloc = self._analyses[construction]
        self._modify_construction_count(construction, -count)
        return count

    def _random_split(self, compound, threshold):
        """Return a random split for compound.

        Arguments:
            compound: compound to split
            threshold: probability of splitting at each position

        """
        forced = segmentation_to_splitloc(self._force_split(compound))
        splitloc = tuple(i for i in range(1, len(compound))
                         if (i in forced or random.random() < threshold))
        if self._restricted and compound in self.allowed_boundaries:
            allowed = self.allowed_boundaries[compound]
            splitloc = tuple(i for i in splitloc if i in allowed)
        return splitloc_to_segmentation(compound, splitloc)

    def _clear_compound_analysis(self, compound):
        """Clear analysis of a compound from model"""
        assert(compound in self._compounds)
        rcount, splitloc = self._compounds[compound]
        if splitloc:
            for child in splitloc_to_segmentation(compound, splitloc):
                self._modify_construction_count(child, -rcount)
        else:
            self._modify_construction_count(compound, -rcount)

    def _set_compound_analysis(self, compound, parts):
        """Set analysis of compound to according to given segmentation.

        Arguments:
            compound: compound to split
            parts: desired constructions of the compound

        Note: _clear_compound_analysis should usually be called first.

        """
        assert(compound in self._compounds)
        rcount = self._compounds[compound].count
        if len(parts) == 1:
            self._compounds[compound] = \
                self._compounds[compound]._replace(splitloc=tuple())
            self._modify_construction_count(compound, rcount)
        else:
            self._compounds[compound] = \
                self._compounds[compound]._replace(
                splitloc=segmentation_to_splitloc(parts))
            for constr in parts:
                self._modify_construction_count(constr, rcount)

    def get_construction_count(self, construction):
        """Return (real) count of the construction."""
        if (construction in self._analyses and
            not self._analyses[construction].splitloc):
            count = self._analyses[construction].count
            if count <= 0:
                raise MorfessorException("Construction count of '%s' is %s"
                                         % (construction, count))
            return count
        return 0

    def _update_annotation_choices(self):
        """Update the selection of alternative analyses in annotations.

        For semi-supervised models, select the most likely alternative
        analyses included in the annotations of the compounds.

        """
        if not self._supervised:
            return

        # Collect constructions from the most probable segmentations
        # and add missing compounds also to the unannotated data
        constructions = collections.Counter()
        for compound, alternatives in self.annotations.items():
            if not compound in self._compounds:
                self._add_compound(compound, 1)

            analysis, cost = self._best_analysis(alternatives)
            for constr in analysis:
                constructions[constr] += self._compounds[compound].count

        # Apply the selected constructions in annotated corpus coding
        self._annot_coding.set_constructions(constructions)
        for constr in constructions.keys():
            count = self.get_construction_count(constr)
            self._annot_coding.set_count(constr, count)

    def _best_analysis(self, choices):
        """Select the best analysis out of the given choices."""
        bestcost = None
        bestanalysis = None
        for analysis in choices:
            cost = 0.0
            for constr in analysis:
                count = self.get_construction_count(constr)
                if count > 0:
                    cost += (math.log(self._corpus_coding.tokens) -
                             math.log(count))
                else:
                    cost -= self.penalty  # penalty is negative
            if bestcost is None or cost < bestcost:
                bestcost = cost
                bestanalysis = analysis
        return bestanalysis, bestcost

    def _force_split(self, compound):
        """Return forced split of the compound."""
        if len(self.forcesplit_list) == 0:
            return [compound]
        clen = len(compound)
        if self._restricted and compound in self.allowed_boundaries:
            indices = self.allowed_boundaries[compound]
        else:
            indices = range(1, clen)
        j = 0
        parts = []
        for i in indices:
            if compound[i] in self.forcesplit_list:
                if len(compound[j:i]) > 0:
                    parts.append(compound[j:i])
                parts.append(compound[i:i + 1])
                j = i + 1
        if j < clen:
            parts.append(compound[j:])
        return [p for p in parts if len(p) > 0]

    def _test_skip(self, construction):
        """Return true if construction should be skipped."""
        if construction in self._counter:
            t = self._counter[construction]
            if random.random() > 1.0 / max(1, t):
                return True
        self._counter[construction] += 1
        return False

    def _viterbi_optimize(self, compound, addcount=0, maxlen=30):
        """Optimize segmentation of the compound using the Viterbi algorithm.

        Arguments:
          compound: compound to optimize
          addcount: constant for additive smoothing of Viterbi probs
          maxlen: maximum length for a construction

        Returns list of segments.

        """
        clen = len(compound)
        if clen == 1:  # Single atom
            return [compound]
        if self._use_skips and self._test_skip(compound):
            return self.segment(compound)

        # Remove old analysis
        self._clear_compound_analysis(compound)

        # Restricted boundaries
        if self._restricted and compound in self.allowed_boundaries:
            allowed = self.allowed_boundaries[compound]
        else:
            allowed = None # Everything is allowed

        # Collect forced subsegments
        parts = self._force_split(compound)
        if len(parts) > 1:
            constructions = []
            pos = 0
            for part in parts:
                if allowed is None:
                    constructions += self.viterbi_segment(
                        part, addcount=addcount, maxlen=maxlen)[0]
                else:
                    allowed_in_part = [x-pos for x in allowed
                                       if (x > pos and x < pos+len(part))]
                    constructions += self.viterbi_segment(
                        part, addcount=addcount, maxlen=maxlen,
                        allowed_boundaries=allowed_in_part)[0]
                pos += len(part)
        else:
            constructions = self.viterbi_segment(
                compound, addcount=addcount, maxlen=maxlen,
                allowed_boundaries=allowed)[0]

        # Set new analysis
        self._set_compound_analysis(compound, constructions)
        return constructions

    def _recursive_optimize(self, compound):
        """Optimize segmentation of the compound using recursive splitting.

        Returns list of segments.

        """
        if len(compound) == 1:  # Single atom
            return [compound]
        if self._use_skips and self._test_skip(compound):
            return self.segment(compound)

        # Restricted boundaries
        if self._restricted and compound in self.allowed_boundaries:
            allowed = self.allowed_boundaries[compound]
        else:
            allowed = None # Everything is allowed

        # Collect forced subsegments
        parts = self._force_split(compound)
        if len(parts) > 1:
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, parts)
            # Use recursive algorithm to optimize the subsegments
            constructions = []
            pos = 0
            for part in parts:
                if allowed is None:
                    constructions += self._recursive_split(part)
                else:
                    allowed_in_part = [x-pos for x in allowed
                                       if (x > pos and x < pos+len(part))]
                    constructions += self._recursive_split(
                        part, allowed_in_part)
                pos += len(part)
        else:
            constructions = self._recursive_split(compound, allowed)

        return constructions

    def _recursive_split(self, construction, allowed_boundaries=None):
        """Optimize segmentation of the construction by recursive splitting.

        Returns list of segments.

        """
        if allowed_boundaries is not None:
            _logger.debug("restricted: %s %s", construction,
                          allowed_boundaries)
        if len(construction) == 1:  # Single atom
            return [construction]
        if self._use_skips and self._test_skip(construction):
            return self._segment(construction)
        count = self._remove(construction)

        # Check all binary splits and no split
        self._modify_construction_count(construction, count)
        mincost = self.get_cost()
        self._modify_construction_count(construction, -count)
        splitloc = 0
        indices = range(1, len(construction)) \
            if allowed_boundaries is None else allowed_boundaries
        for i in indices:
            assert(i < len(construction))
            if (self.nosplit_re and
                self.nosplit_re.match(construction[(i - 1):(i + 1)])):
                continue
            prefix = construction[:i]
            suffix = construction[i:]
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            cost = self.get_cost()
            self._modify_construction_count(prefix, -count)
            self._modify_construction_count(suffix, -count)
            if cost <= mincost:
                mincost = cost
                splitloc = i

        if splitloc:
            # Virtual construction
            self._analyses[construction] = ConstrNode(count, (splitloc,))
            prefix = construction[:splitloc]
            suffix = construction[splitloc:]
            self._modify_construction_count(prefix, count)
            self._modify_construction_count(suffix, count)
            if allowed_boundaries is None:
                lp = self._recursive_split(prefix)
                if suffix != prefix:
                    return lp + self._recursive_split(suffix)
                else:
                    return lp + lp
            else:
                lp = self._recursive_split(
                    prefix,
                    [x for x in allowed_boundaries if x < splitloc])
                ls = self._recursive_split(
                    suffix,
                    [x-splitloc for x in allowed_boundaries if x > splitloc])
                return lp + ls
        else:
            # Real construction
            self._analyses[construction] = ConstrNode(0, tuple())
            self._modify_construction_count(construction, count)
            return [construction]

    def _modify_construction_count(self, construction, dcount):
        """Modify the count of construction by dcount.

        For virtual constructions, recurses to child nodes in the
        tree. For real constructions, adds/removes construction
        to/from the lexicon whenever necessary.

        """
        if construction in self._analyses:
            count, splitloc = self._analyses[construction]
        else:
            count, splitloc = 0, tuple()
        newcount = count + dcount
        if newcount == 0:
            del self._analyses[construction]
        else:
            self._analyses[construction] = ConstrNode(newcount, splitloc)
        if splitloc:
            # Virtual construction
            children = splitloc_to_segmentation(construction, splitloc)
            for child in children:
                self._modify_construction_count(child, dcount)
        else:
            # Real construction
            #_logger.debug("Updating corpus count: %s -> %s", count, newcount)
            self._corpus_coding.update_count(construction, count, newcount)
            if self._supervised:
                self._annot_coding.update_count(construction, count, newcount)

            if count == 0 and newcount > 0:
                #_logger.debug("Adding to lex: %s", construction)
                self._lexicon_coding.add(construction)
            elif count > 0 and newcount == 0:
                #_logger.debug("Removing from lex: %s", construction)
                self._lexicon_coding.remove(construction)

    def _epoch_update(self, epoch_num):
        """Do model updates that are necessary between training epochs.

        The argument is the number of training epochs finished.

        In practice, this does two things:
        - If random skipping is in use, reset construction counters.
        - If semi-supervised learning is in use and there are alternative
          analyses in the annotated data, select the annotations that are
          most likely given the model parameters. If not hand-set, update
          the weight of the annotated corpus.

        This method should also be run prior to training (with the
        epoch number argument as 0).

        """
        forced_epochs = 0
        if self._corpus_weight_updater.update(self, epoch_num):
            forced_epochs += 2

        if self._use_skips:
            self._counter = collections.Counter()
        if self._supervised:
            self._update_annotation_choices()
            self._annot_coding.update_weight()

        return forced_epochs

    def get_compounds(self):
        """Return the compound types stored by the model."""
        self._check_segment_only()
        return self._compounds.keys()

    def get_constructions(self):
        """Return a list of the present constructions and their counts."""
        return sorted((c, node.count) for c, node in self._analyses.items()
                      if not node.splitloc)

    def get_cost(self):
        """Return current model encoding cost."""
        cost = self._corpus_coding.get_cost() + self._lexicon_coding.get_cost()
        if self._supervised:
            return cost + self._annot_coding.get_cost()
        else:
            return cost

    def get_segmentations(self):
        """Retrieve segmentations for all compounds encoded by the model."""
        self._check_segment_only()
        for compound in sorted(self._compounds.keys()):
            count = self._compounds[compound].count
            yield count, compound, self.segment(compound)

    def load_data(self, data, freqthreshold=1, count_modifier=None,
                  init_rand_split=None):
        """Load data to initialize the model for batch training.

        Arguments:
            data: iterator of (count, compound_atoms) tuples
            freqthreshold: discard compounds that occur less than
                             given times in the corpus (default 1)
            count_modifier: function for adjusting the counts of each
                              compound
            init_rand_split: If given, random split the word with
                               init_rand_split as the probability for each
                               split

        Adds the compounds in the corpus to the model lexicon. Returns
        the total cost.

        """
        self._check_segment_only()
        totalcount = collections.Counter()
        for count, atoms in data:
            if len(atoms) > 0:
                totalcount[atoms] += count

        for atoms, count in totalcount.items():
            if count < freqthreshold:
                continue
            if count_modifier is not None:
                self._add_compound(atoms, count_modifier(count))
            else:
                self._add_compound(atoms, count)

            if init_rand_split is not None and init_rand_split > 0:
                parts = self._random_split(atoms, init_rand_split)
                self._clear_compound_analysis(atoms)
                self._set_compound_analysis(atoms, parts)

        return self.get_cost()

    def load_segmentations(self, segmentations):
        """Load model from existing segmentations.

        The argument should be an iterator providing a count, a
        compound, and its segmentation.

        """
        self._check_segment_only()
        for count, compound, segmentation in segmentations:
            self._add_compound(compound, count)
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, segmentation)

    def set_annotations(self, annotations, annotatedcorpusweight=None):
        """Prepare model for semi-supervised learning with given
         annotations.

         """
        self._check_segment_only()
        self._supervised = True
        self.annotations = annotations
        self._annot_coding = AnnotatedCorpusEncoding(
            self._corpus_coding, weight= annotatedcorpusweight)
        self._annot_coding.boundaries = len(self.annotations)

    def set_restrictions(self, annotations, relaxed=None):
        """Set segmentation restrictions from annotations.

        Arguments:
            annotations: iterator over annotations
            relaxed: use relaxed restrictions with tuple of at most
                       four integers (left, right, begin, end)

        In relaxed restrictions, positions next to actual segmentation
        points in annotations are allowed. The numbers "left" and
        "right" give the range of allowed positions left and right of
        the correct segmentation point. Begin and end give the range
        of allowed positions in the beginning and end of the compound.

        """
        self._check_segment_only()
        self._restricted = True
        if relaxed:
            if isinstance(relaxed, tuple) and len(relaxed) < 5:
                left, right, begin, end = relaxed + (0,)*(4-len(relaxed))
            else:
                raise MorfessorException("Improper value for relaxed: %s"
                                         % repr(relaxed))
        else:
            left, right, begin, end = (0, 0, 0, 0)
        self.allowed_boundaries = {}
        for compound in annotations:
            allowed_positions = set()
            for analysis in annotations[compound]:
                for idx in segmentation_to_splitloc(analysis):
                    assert(idx < len(compound))
                    allowed_positions.add(idx)
                    for ctx in range(left):
                        if idx-ctx-1 > 0:
                            allowed_positions.add(idx-ctx-1)
                    for ctx in range(right):
                        if idx+ctx+1 < len(compound):
                            allowed_positions.add(idx+ctx+1)
            for ctx in range(begin):
                if ctx+1 < len(compound):
                    allowed_positions.add(ctx+1)
            for ctx in range(end):
                if ctx+1 < len(compound):
                    allowed_positions.add(len(compound)-ctx-1)
            self.allowed_boundaries[compound] = sorted(allowed_positions)

    def _segment(self, construction):
        """Return real constructions of a potentially virtual construction.

        Makes a recursive call on children of virtual constructions.

        """
        count, splitloc = self._analyses[construction]
        constructions = []
        if splitloc:
            for child in splitloc_to_segmentation(construction, splitloc):
                constructions += self._segment(child)
        else:
            constructions.append(construction)
        return constructions

    def segment(self, compound):
        """Segment the compound by looking it up in the model analyses.

        Raises KeyError if compound is not present in the training
        data. For segmenting new words, use viterbi_segment(compound).

        """
        self._check_segment_only()
        count, splitloc = self._compounds[compound]
        constructions = []
        if splitloc:
            children = splitloc_to_segmentation(compound, splitloc)
        else:
            children = [compound]
        for child in children:
            constructions += self._segment(child)
        return constructions

    def train_batch(self, algorithm='recursive', algorithm_params=(),
                    finish_threshold=0.005, max_epochs=None):
        """Train the model in batch fashion.

        The model is trained with the data already loaded into the model (by
        using an existing model or calling one of the load\_ methods).

        In each iteration (epoch) all compounds in the training data are
        optimized once, in a random order. If applicable, corpus weight,
        annotation cost, and random split counters are recalculated after
        each iteration.

        Arguments:
            algorithm: string in ('recursive', 'viterbi', 'flatten')
                         that indicates the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            finish_threshold: the stopping threshold. Training stops when
                                the improvement of the last iteration is
                                smaller then finish_threshold * #boundaries
            max_epochs: maximum number of epochs to train

        """
        epochs = 0
        forced_epochs = max(1, self._epoch_update(epochs))
        newcost = self.get_cost()
        compounds = list(self.get_compounds())
        _logger.info("Compounds in training data: %s types / %s tokens" %
                     (len(compounds), self._corpus_coding.boundaries))

        if algorithm == 'flatten':
            _logger.info("Flattening analysis tree")
            for compound in _progress(compounds):
                parts = self.segment(compound)
                self._clear_compound_analysis(compound)
                self._set_compound_analysis(compound, parts)
            _logger.info("Done.")
            return 1, self.get_cost()

        _logger.info("Starting batch training")
        _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))
        self._check_integrity()

        while True:
            # One epoch
            random.shuffle(compounds)

            for w in _progress(compounds):
                if algorithm == 'recursive':
                    segments = self._recursive_optimize(w, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(w, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                _logger.debug("#%s -> %s" %
                              (w, constructions_to_str(segments)))

            self._check_integrity()
            epochs += 1

            _logger.debug("Cost before epoch update: %s" % self.get_cost())
            forced_epochs = max(forced_epochs, self._epoch_update(epochs))
            oldcost = newcost
            newcost = self.get_cost()

            self._check_restrictions()
            _logger.info("Epochs: %s\tCost: %s" % (epochs, newcost))
            if (forced_epochs == 0 and
                    newcost >= oldcost - finish_threshold *
                    self._corpus_coding.boundaries):
                break
            if forced_epochs > 0:
                forced_epochs -= 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break
        _logger.info("Done.")
        return epochs, newcost

    def train_online(self, data, count_modifier=None, epoch_interval=10000,
                     algorithm='recursive', algorithm_params=(),
                     init_rand_split=None, max_epochs=None):
        """Train the model in online fashion.

        The model is trained with the data provided in the data argument.
        As example the data could come from a generator linked to standard in
        for live monitoring of the splitting.

        All compounds from data are only optimized once. After online
        training, batch training could be used for further optimization.

        Epochs are defined as a fixed number of compounds. After each
        epoch (like in batch training), the annotation cost, and
        random split counters are recalculated if applicable.

        Arguments:
            data: iterator of (_, compound_atoms) tuples. The first
                    argument is ignored, as every occurence of the
                    compound is taken with count 1
            count_modifier: function for adjusting the counts of each
                              compound
            epoch_interval: number of compounds to process before starting
                              a new epoch
            algorithm: string in ('recursive', 'viterbi') that indicates
                         the splitting algorithm used.
            algorithm_params: parameters passed to the splitting algorithm.
            init_rand_split: probability for random splitting a compound to
                               at any point for initializing the model. None
                               or 0 means no random splitting.
            max_epochs: maximum number of epochs to train

        """
        self._check_segment_only()
        if count_modifier is not None:
            counts = {}

        _logger.info("Starting online training")

        epochs = 0
        i = 0
        more_tokens = True
        while more_tokens:
            self._epoch_update(epochs)
            newcost = self.get_cost()
            _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))

            for _ in _progress(range(epoch_interval)):
                try:
                    _, w = next(data)
                except StopIteration:
                    more_tokens = False
                    break

                if len(w) == 0:
                    # Newline in corpus
                    continue

                if count_modifier is not None:
                    if not w in counts:
                        c = 0
                        counts[w] = 1
                        addc = 1
                    else:
                        c = counts[w]
                        counts[w] = c + 1
                        addc = count_modifier(c + 1) - count_modifier(c)
                    if addc > 0:
                        self._add_compound(w, addc)
                else:
                    self._add_compound(w, 1)
                if init_rand_split is not None and init_rand_split > 0:
                    parts = self._random_split(w, init_rand_split)
                    self._clear_compound_analysis(w)
                    self._set_compound_analysis(w, parts)
                if algorithm == 'recursive':
                    segments = self._recursive_optimize(w, *algorithm_params)
                elif algorithm == 'viterbi':
                    segments = self._viterbi_optimize(w, *algorithm_params)
                else:
                    raise MorfessorException("unknown algorithm '%s'" %
                                             algorithm)
                _logger.debug("#%s: %s -> %s" %
                              (i, w, constructions_to_str(segments)))
                i += 1

            epochs += 1
            if max_epochs is not None and epochs >= max_epochs:
                _logger.info("Max number of epochs reached, stop training")
                break

        self._epoch_update(epochs)
        newcost = self.get_cost()
        _logger.info("Tokens processed: %s\tCost: %s" % (i, newcost))
        return epochs, newcost

    def viterbi_segment(self, compound, addcount=1.0, maxlen=30,
                        allowed_boundaries=None):
        """Find optimal segmentation using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the most probable segmentation and its log-probability.

        """
        clen = len(compound)
        indices = range(1, clen+1) if allowed_boundaries is None \
                  else allowed_boundaries+[clen]
        grid = [(0.0, None)]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries + \
                addcount > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries + addcount)
        else:
            logtokens = 0
        if addcount > 0:
            newboundcost = (self._lexicon_coding.boundaries + addcount) * \
                           math.log(self._lexicon_coding.boundaries + addcount)
            if self._lexicon_coding.boundaries > 0:
                newboundcost -= self._lexicon_coding.boundaries * \
                                math.log(self._lexicon_coding.boundaries)
            newboundcost /= self._corpus_coding.weight
        else:
            newboundcost = 0
        badlikelihood = 1.0 + clen * logtokens + newboundcost + \
                        self._lexicon_coding.get_codelength(compound) / \
                        self._corpus_coding.weight
        # Viterbi main loop
        for t in range(1, clen + 1):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestpath = None
            bestcost = None
            if ((self.nosplit_re and t < clen and
                 self.nosplit_re.match(compound[(t-1):(t+1)])) or
                (t < clen and not t in indices)):
                grid.append((clen*badlikelihood, t-1))
                continue
            for pt in range(max(0, t - maxlen), t):
                if grid[pt][0] is None:
                    continue
                cost = grid[pt][0]
                construction = compound[pt:t]
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - math.log(count + addcount))
                elif addcount > 0:
                    if self._corpus_coding.tokens == 0:
                        cost += (addcount * math.log(addcount) +
                                 newboundcost +
                                 self._lexicon_coding.get_codelength(
                                     construction) /
                                 self._corpus_coding.weight)
                    else:
                        cost += (logtokens - math.log(addcount) +
                                 newboundcost +
                                 self._lexicon_coding.get_codelength(
                                     construction) /
                                 self._corpus_coding.weight)
                elif len(construction) == 1:
                    cost += badlikelihood
                elif self.nosplit_re or allowed_boundaries is not None:
                    # Some splits are forbidden, so longer unknown
                    # constructions have to be allowed
                    cost += len(construction) * badlikelihood
                else:
                    continue
                #_logger.debug("cost(%s)=%.2f", construction, cost)
                if bestcost is None or cost < bestcost:
                    bestcost = cost
                    bestpath = pt
            grid.append((bestcost, bestpath))
        constructions = []
        cost, path = grid[-1]
        lt = clen + 1
        while path is not None:
            t = path
            constructions.append(compound[t:lt])
            path = grid[t][1]
            lt = t
        constructions.reverse()
        # Add boundary cost
        cost += (math.log(self._corpus_coding.tokens +
                          self._corpus_coding.boundaries) -
                 math.log(self._corpus_coding.boundaries))
        return constructions, cost

    def forward_logprob(self, compound):
        """Find log-probability of a compound using the forward algorithm.

        Arguments:
          compound: compound to process

        Returns the (negative) log-probability of the compound. If the
        probability is zero, returns a number that is larger than the
        value defined by the penalty attribute of the model object.

        """
        clen = len(compound)
        grid = [0.0]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries)
        else:
            logtokens = 0
        # Forward main loop
        for t in range(1, clen + 1):
            # Sum probabilities from all paths to the current node.
            # Note that we can come from any node in history.
            psum = 0.0
            for pt in range(0, t):
                cost = grid[pt]
                construction = compound[pt:t]
                count = self.get_construction_count(construction)
                if count > 0:
                    cost += (logtokens - math.log(count))
                else:
                    continue
                psum += math.exp(-cost)
            if psum > 0:
                grid.append(-math.log(psum))
            else:
                grid.append(-self.penalty)
        cost = grid[-1]
        # Add boundary cost
        cost += (math.log(self._corpus_coding.tokens +
                          self._corpus_coding.boundaries) -
                 math.log(self._corpus_coding.boundaries))
        return cost

    def viterbi_nbest(self, compound, n, addcount=1.0, maxlen=30):
        """Find top-n optimal segmentations using the Viterbi algorithm.

        Arguments:
          compound: compound to be segmented
          n: how many segmentations to return
          addcount: constant for additive smoothing (0 = no smoothing)
          maxlen: maximum length for the constructions

        If additive smoothing is applied, new complex construction types can
        be selected during the search. Without smoothing, only new
        single-atom constructions can be selected.

        Returns the n most probable segmentations and their
        log-probabilities.

        """
        clen = len(compound)
        grid = [[(0.0, None, None)]]
        if self._corpus_coding.tokens + self._corpus_coding.boundaries + \
                addcount > 0:
            logtokens = math.log(self._corpus_coding.tokens +
                                 self._corpus_coding.boundaries + addcount)
        else:
            logtokens = 0
        if addcount > 0:
            newboundcost = (self._lexicon_coding.boundaries + addcount) * \
                           math.log(self._lexicon_coding.boundaries + addcount)
            if self._lexicon_coding.boundaries > 0:
                newboundcost -= self._lexicon_coding.boundaries * \
                                math.log(self._lexicon_coding.boundaries)
            newboundcost /= self._corpus_coding.weight
        else:
            newboundcost = 0
        badlikelihood = 1.0 + clen * logtokens + newboundcost + \
                        self._lexicon_coding.get_codelength(compound) / \
                        self._corpus_coding.weight
        # Viterbi main loop
        for t in range(1, clen + 1):
            # Select the best path to current node.
            # Note that we can come from any node in history.
            bestn = []
            if self.nosplit_re and t < clen and \
                    self.nosplit_re.match(compound[(t-1):(t+1)]):
                grid.append([(-clen*badlikelihood, t-1, -1)])
                continue
            for pt in range(max(0, t - maxlen), t):
                for k in range(len(grid[pt])):
                    if grid[pt][k][0] is None:
                        continue
                    cost = grid[pt][k][0]
                    construction = compound[pt:t]
                    count = self.get_construction_count(construction)
                    if count > 0:
                        cost += (logtokens - math.log(count + addcount))
                    elif addcount > 0:
                        if self._corpus_coding.tokens == 0:
                            cost -= (addcount * math.log(addcount) +
                                     newboundcost +
                                     self._lexicon_coding.get_codelength(
                                         construction)
                                     / self._corpus_coding.weight)
                        else:
                            cost -= (logtokens - math.log(addcount) +
                                     newboundcost +
                                     self._lexicon_coding.get_codelength(
                                         construction)
                                     / self._corpus_coding.weight)
                    elif len(construction) == 1:
                        cost -= badlikelihood
                    elif self.nosplit_re:
                        # Some splits are forbidden, so longer unknown
                        # constructions have to be allowed
                        cost -= len(construction) * badlikelihood
                    else:
                        continue
                    if len(bestn) < n:
                        heapq.heappush(bestn, (cost, pt, k))
                    else:
                        heapq.heappushpop(bestn, (cost, pt, k))
            grid.append(bestn)
        results = []
        for k in range(len(grid[-1])):
            constructions = []
            cost, path, ki = grid[-1][k]
            lt = clen + 1
            while path is not None:
                t = path
                constructions.append(compound[t:lt])
                path = grid[t][ki][1]
                ki = grid[t][ki][2]
                lt = t
            constructions.reverse()
            # Add boundary cost
            cost -= (math.log(self._corpus_coding.tokens +
                              self._corpus_coding.boundaries) -
                     math.log(self._corpus_coding.boundaries))
            results.append((-cost, constructions))
        return [(constr, cost) for cost, constr in sorted(results)]

    def get_corpus_coding_weight(self):
        return self._corpus_coding.weight

    def set_corpus_coding_weight(self, weight):
        self._check_segment_only()
        self._corpus_coding.weight = weight

    def make_segment_only(self):
        """Reduce the size of this model by removing all non-morphs from the
        analyses. After calling this method it is not possible anymore to call
        any other method that would change the state of the model. Anyway
        doing so would throw an exception.

        """
        self._num_compounds = len(self.get_compounds())
        self._segment_only = True

        self._analyses = {k: v for (k, v) in self._analyses.items()
                          if not v.splitloc}
        self._compounds = {}

    def clear_segmentation(self):
        for compound in self.get_compounds():
            self._clear_compound_analysis(compound)
            self._set_compound_analysis(compound, [compound])

