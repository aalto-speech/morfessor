Command line tools
==================

The installation process installs 4 scripts in the appropriate PATH.

morfessor
---------
The morfessor command is a full-featured script for training, updating models
and segmenting test data.

morfessor-train
---------------
The morfessor-train command is a convenience command that enables easier
training for morfessor models.

The basic command structure is: ::

    morfessor-train [arguments] traindata-file [traindata-file ...]

The arguments are identical to the ones for the `morfessor`_ command. The most relevant are:

========= ==============================
-s <file> save binary model
-S <file> save Morfessor 1.0 style model
========= ==============================



morfessor-segment
-----------------
The morfessor-segment command is a convenience command that enables easier
segmentation of test data with a morfessor model.

The basic command structure is: ::

    morfessor-segment [arguments] testcorpus-file [testcorpus-file ...]

The arguments are identical to the ones for the `morfessor`_ command. The most relevant are:

========= ==============================
-l <file> load binary model
-L <file> load Morfessor 1.0 style model
========= ==============================


morfessor-evaluate
------------------
The morfessor-evaluate command is used for evaluating a morfessor model against
a gold-standard. If multiple models are evaluated, it reports statistical
significant differences between them.


Universal command line options
==============================

================ == =====================
--verbose <int>  -v verbose level; controls what is written to the standard error stream or log file (default 1)
--logfile <file>    write log messages to file in addition to standard error stream
--progressbar       Force the progressbar to be displayed (possibly lowers the log level for the standard error stream)
--help           -h show this help message and exit
--version           show version number and exit
================ == =====================