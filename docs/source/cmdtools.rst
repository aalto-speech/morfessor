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
    load binary model
``-L <file>``
    load Morfessor 1.0 style model


Loading data
~~~~~~~~~~~~

``--traindata <file>``
    input corpus file(s) for training (text or bz2/gzipped text; use '-'
    for standard input; add several times in order to append multiple files)
``--traindata-list``
    input file(s) for batch training are lists (one compound per line, optionally count as a prefix)
``--testdata <file> \-T``
    input corpus file(s) to analyze (text or bz2/gzipped text; use '-' for standard input; add several times in order to append multiple files)


Training model options
~~~~~~~~~~~~~~~~~~~~~~

``-m <mode>, --mode <mode>``
    training mode ('none', 'init', 'batch', 'init+batch', 'online', or 'online+batch'; default 'init+batch')
``-a <algorithm>, --algorithm <algorithm>``
    algorithm type ('recursive', 'viterbi'; default'recursive')
``-d <type>, --dampening <type>``
    frequency dampening for training data ('none', 'log',
    or 'ones'; default 'none')
``-f <list>, --forcesplit <list>``
    force split on given atoms (default ['-'])
``-F <float>, --finish-threshold <float>``
    Stopping threshold. Training stops when the
    improvement of the last iteration issmaller then
    finish_threshold * #boundaries; (default '0.005')
``-r <seed>, --randseed <seed>``
    seed for random number generator
``-R <float>, --randsplit <float>``
    initialize new words by random splitting using the
    given split probability (default no splitting)
``--skips``
    use random skips for frequently seen compounds to
    speed up training
``--batch-minfreq <int>``
    compound frequency threshold for batch training
    (default 1)
``--max-epochs <int>``
    hard maximum of epochs in training
``--nosplit-re <regexp>``
    if the expression matches the two surrounding
    characters, do not allow splitting (default None)
``--online-epochint <int>``
    epoch interval for online training (default 10000)
``--viterbi-smoothing <float>``
    additive smoothing parameter for Viterbi training and
    segmentation (default 0)
``--viterbi-maxlen <int>``
    maximum construction length in Viterbi training and
    segmentation (default 30)


Saving model
~~~~~~~~~~~~

``-s <file>``
    save binary model
``-S <file>``
    save Morfessor 1.0 style model


morfessor-train
---------------
The morfessor-train command is a convenience command that enables easier
training for morfessor models.

The basic command structure is: ::

    morfessor-train [arguments] traindata-file [traindata-file ...]

The arguments are identical to the ones for the `morfessor`_ command. The most relevant are:

``-s <file>``
    save binary model
``-S <file>``
    save Morfessor 1.0 style model




morfessor-segment
-----------------
The morfessor-segment command is a convenience command that enables easier
segmentation of test data with a morfessor model.

The basic command structure is: ::

    morfessor-segment [arguments] testcorpus-file [testcorpus-file ...]

The arguments are identical to the ones for the `morfessor`_ command. The most relevant are:

``-l <file>``
    load binary model
``-L <file>``
    load Morfessor 1.0 style model


morfessor-evaluate
------------------
The morfessor-evaluate command is used for evaluating a morfessor model against
a gold-standard. If multiple models are evaluated, it reports statistical
significant differences between them.


Data format command line options
--------------------------------


``--encoding <encoding>``
    encoding of input and output files (if none is given, both the local
    encoding and UTF-8 are tried)
``--lowercase``
    lowercase input data
``--traindata-list``
    input file(s) for batch training are lists (one compound per line, optionally count as a prefix)
``--atom-separator <regexp>``
    atom separator regexp (default None)
``--compound-separator <regexp>``
    compound separator regexp (default '\s+')
``--analysis-separator <str>``
    separator for different analyses in an annotation file. Use NONE for only allowing one analysis per line
``--output-format <format>``
    format string for --output file (default: '{analysis}\n'). Valid keywords are: {analysis} = constructions of the compound, {compound} = compound string, {count} = count of the compound (currently always 1), {logprob} = log-probability of the analysis, and {clogprob} = log-probability of the compound. Valid escape sequences are '\n' (newline) and '\t' (tabular)
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
