Command line tools
==================

The installation process installs 4 scripts in the appropriate PATH.

morfessor
---------
The morfessor command is a full-featured script for training, updating models
and segmenting test data.

Loading existing model
~~~~~~~~~~~~~~~~~~~~~~

``-l <file>``
    load :ref:`binary-model-def`
``-L <file>``
    load :ref:`morfessor1-model-def`


Loading data
~~~~~~~~~~~~

``-t <file>, --traindata <file>``
    Input corpus file(s) for training (text or bz2/gzipped text; use '-'
    for standard input; add several times in order to append multiple files).
    Standard, all sentences are split on whitespace and the tokens are used as
    compounds. The ``--traindata-list`` option can be used to read all input
    files as a list of compounds, one compound per line optionally prefixed by
    a count. See :ref:`data-format-options` for changing the delimiters used for
    separating compounds and atoms.
``--traindata-list``
    Interpret all training files as list files instead of corpus files. A list
    file contains one compound per line with optionally a count as prefix.
``-T <file>, --testdata <file>``
    Input corpus file(s) to analyze (text or bz2/gzipped text; use '-' for
    standard input; add several times in order to append multiple files). The
    file is read in the same manner as an input corpus file. See
    :ref:`data-format-options` for changing the delimiters used for
    separating compounds and atoms.


Training model options
~~~~~~~~~~~~~~~~~~~~~~

``-m <mode>, --mode <mode>``
    Morfessor can run in different modes, each doing different actions on the
    model. The modes are:

    none
        Do initialize or train a model. Can be used when just loading a model
        for segmenting new data
    init
        Create new model and load input data. Does not train the model
    batch
        Loads an existing model (which is already initialized with training
        data) and run :ref:`batch-training`
    init+batch
        Create a new model, load input data and run :ref:`batch-training`.
        **Default**
    online
        Create a new model, read and train the model concurrently as described
        in :ref:`online-training`
    online+batch
        First read and train the model concurrently as described in
        :ref:`online-training` and after that retrain the model using
        :ref:`batch-training`


``-a <algorithm>, --algorithm <algorithm>``
    Algorithm to use for training:

    recursive
        Recursive as descirbed in :ref:`recursive-training` **Default**
    viterbi
        Viterbi as described in :ref:`viterbi-training`

``-d <type>, --dampening <type>``
    Method for changing the compound counts in the input data. Options:

    none
        Do not alter the counts of compounds (token based training)
    log
        Change the count :math:`x` of a compound to :math:`\log(x)` (log-token
        based training)
    ones
        Treat all compounds as if they only occured once (type based training)

``-f <list>, --forcesplit <list>``
    A list of atoms that would always cause the compound to be split. By
    default only hyphens (``-``) would force a split. Note the notation of the
    argument list. To have no force split characters, use as an empty string as
    argument (``-f ""``). To split, for example, both hyphen (``-``) and
    apostrophe (``'``) use ``-f "-'"``

``-F <float>, --finish-threshold <float>``
    Stopping threshold. Training stops when the decrease in model cost of the
    last iteration is smaller then finish_threshold * #boundaries; (default
    '0.005')

``-r <seed>, --randseed <seed>``
    Seed for random number generator

``-R <float>, --randsplit <float>``
    Initialize new words by random splitting using the
    given split probability (default no splitting). See :ref:`rand-init`

``--skips``
    Use random skips for frequently seen compounds to
    speed up training. See :ref:`rand-init`

``--batch-minfreq <int>``
    Compound frequency threshold for batch training
    (default 1)
``--max-epochs <int>``
    Hard maximum of epochs in training
``--nosplit-re <regexp>``
    If the expression matches the two surrounding
    characters, do not allow splitting (default None)
``--online-epochint <int>``
    Epoch interval for online training (default 10000)
``--viterbi-smoothing <float>``
    Additive smoothing parameter for Viterbi training and
    segmentation (default 0).
``--viterbi-maxlen <int>``
    Maximum construction length in Viterbi training and
    segmentation (default 30)


Saving model
~~~~~~~~~~~~

``-s <file>``
    save  :ref:`binary-model-def`
``-S <file>``
    save  :ref:`morfessor1-model-def`
``--save-reduced``
    save :ref:`binary-reduced-model-def`

Examples
~~~~~~~~
Training a model from inputdata.txt, saving a :ref:`morfessor1-model-def` and
segmenting the test.txt set: ::

    morfessor -t inputdata.txt -S model.segm -T test.txt

morfessor-train
---------------
The morfessor-train command is a convenience command that enables easier
training for morfessor models.

The basic command structure is: ::

    morfessor-train [arguments] traindata-file [traindata-file ...]

The arguments are identical to the ones for the `morfessor`_ command. The most
relevant are:

``-s <file>``
    save binary model
``-S <file>``
    save Morfessor 1.0 style model
``--save-reduced``
    save reduced binary model

Examples
~~~~~~~~
Train a morfessor model from a wordcount list in ISO_8859-15, doing type based
training, writing the log to logfile and saving them model as model.bin: ::

    morfessor-train --encoding=ISO_8859-15 --traindata-list --logfile=log.log -s model.bin -d ones traindata.txt

morfessor-segment
-----------------
The morfessor-segment command is a convenience command that enables easier
segmentation of test data with a morfessor model.

The basic command structure is: ::

    morfessor-segment [arguments] testcorpus-file [testcorpus-file ...]

The arguments are identical to the ones for the `morfessor`_ command. The most
 relevant are:

``-l <file>``
    load binary model (normal or reduced)
``-L <file>``
    load Morfessor 1.0 style model

Examples
~~~~~~~~
Loading a binary model and segmenting the words in testdata.txt: ::

    morfessor-segment -l model.bin testdata.txt

morfessor-evaluate
------------------
The morfessor-evaluate command is used for evaluating a morfessor model against
a gold-standard. If multiple models are evaluated, it reports statistical
significant differences between them.

The basic command structure is: ::

    morfessor-evaluate [arguments] <goldstandard> <model> [<model> ...]


Positional arguments
~~~~~~~~~~~~~~~~~~~~
``<goldstandard>``
    gold standard file in standard annotation format
``<model>``
    model files to segment (either binary or Morfessor 1.0 style segmentation
    models).

Optional arguments
~~~~~~~~~~~~~~~~~~
``-t TEST_SEGMENTATIONS, --testsegmentation TEST_SEGMENTATIONS``
    Segmentation of the test set. Note that all words in the gold-standard must
     be segmented

``--num-samples <int>``
    number of samples to take for testing
``--sample-size <int>``
    size of each testing samples
``--format-string <format>``
    Python new style format string used to report evaluation results. The
    following variables are a value and and action separated with and
    underscore. E.g. fscore_avg for the average f-score. The available
    values are "precision", "recall", "fscore", "samplesize" and the available
    actions: "avg", "max", "min", "values", "count". A last meta-data variable
    (without action) is "name", the filename of the model. See also the
    format-template option for predefined strings.
``--format-template <template>``
    Uses a template string for the format-string options. Available templates
    are: default, table and latex. If format-string is defined this option is
    ignored.

Examples
~~~~~~~~

Evaluating three different models against a golden standard, outputting the
results in latex table format:::

    morfessor-evaluate --format-template=latex goldstd.txt model1.bin model2.segm model3.bin

.. _data-format-options:

Data format command line options
--------------------------------


``--encoding <encoding>``
    Encoding of input and output files (if none is given, both the local
    encoding and UTF-8 are tried).
``--lowercase``
    lowercase input data
``--traindata-list``
    input file(s) for batch training are lists (one compound per line,
    optionally count as a prefix)
``--atom-separator <regexp>``
    atom separator regexp (default None)
``--compound-separator <regexp>``
    compound separator regexp (default '\s+')
``--analysis-separator <str>``
    separator for different analyses in an annotation file. Use NONE for only
    allowing one analysis per line
``--output-format <format>``
    format string for --output file (default: '{analysis}\\n'). Valid keywords
    are: ``{analysis}`` = constructions of the compound, ``{compound}`` =
    compound string, {count} = count of the compound (currently always 1),
    ``{logprob}`` = log-probability of the analysis, and ``{clogprob}`` =
    log-probability of the compound. Valid escape sequences are ``\n`` (newline)
    and ``\t`` (tabular)
``--output-format-separator <str>``
    construction separator for analysis in --output file (default: ' ')
``--output-newlines``
    for each newline in input, print newline in --output file (default: 'False')




Universal command line options
------------------------------
``--verbose <int>  -v``
    verbose level; controls what is written to the standard error stream or log file (default 1)
``--logfile <file>``
    write log messages to file in addition to standard error stream
``--progressbar``
    Force the progressbar to be displayed (possibly lowers the log level for the standard error stream)
``--help``
    -h show this help message and exit
``--version``
    show version number and exit



Morfessor features
==================

All features below are described in a short format, mainly to guide making the
right choice for a certain parameter. These features are explained in detail in
the :ref:`morfessor-tech-report`.


.. _`batch-training`:

Batch training
--------------
In batch training, each epoch consists of an iteration over the full training
data. Epochs are repeated until the model cost is converged. All training data
needed in the training needs to be loaded before the training starts.

.. _`online-training`:

Online training
---------------
In online training the model is updated while the data is being added. This
allows for rapid testing and prototyping. All data is only processed once,
hence it is advisable to run :ref:`batch-training` afterwards. The size of an
epoch is a fixed, predefined number of compounds processed. The only use of an
epoch for online training is to select the best annotations in semi-supervised
training.

.. _`recursive-training`:

Recursive training
------------------
In recursive training, each compound is processed in the following manner. The
current split for the compound is removed from the model and its constructions
are updated accordingly. After this, all possible splits are tried, by choosing
one split and running the algorithm recursively on the created constructions.

In the end, the best split is selected and the training continues with the next
compound.

.. _`viterbi-training`:

Local Viterbi training
----------------------
In Local Viterbi training the compounds are processed sequentially. Each
compound is removed from the corpus and afterwards segmented using Viterbi
segmentation. The result is put back into the model.

In order to allow new constructions to be created, the smoothing parameter
must be given some non-zero value.

.. _`rand-skips`:

Random skips
------------
In Random skips, frequently seen compounds are skipped in training with a
random probability. As shown in the :ref:`morfessor-tech-report` this speeds
up the training considerably with only a minor loss in model performance.

.. _`rand-init`:

Random initialization
---------------------
In random initialization all compounds are split randomly. Each possible
boundary is made a split with the given probability.

Selecting a good random initialization parameter helps in finding local optima
as long as the split probability is high enough.

.. _`corpusweight`:

Corpusweight (alpha) tuning
---------------------------
An important parameter of the Morfessor Baseline model is the corpusweight
(:math:`\alpha`), which balances the cost of the lexicon and the corpus. There
are different options available for tuning this weight:

Fixed weight (``--corpusweight``)
    The weight is set fixed on the beginning of the training and does not change
Development set (``--develset``)
    A development set is used to balance the corpusweight so that the precision
    and recall of segmenting the developmentset will be equal
Morph length (``--morph-length``)
    The corpusweight is tuned so that the average length of morphs in the
    lexicon will be as desired
Num morph types (``--num-morph-types``)
    The corpusweight is tuned so that there will be approximate the number of
    desired morph types in the lexicon
