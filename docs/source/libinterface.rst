Python library interface to Morfessor
=====================================

Morfessor 2.0 contains a library interface in order to be integrated in other
python applications. The public members are documented below and should remain
relatively the same between Morfessor versions. Private members are documented
in the code and can change anytime in releases.

The classes are documented below.

IO class
--------
.. automodule:: morfessor.io
   :members:

.. _baseline-model-label:

Model classes
-------------
.. automodule:: morfessor.baseline
   :members:

Evaluation classes
------------------
.. automodule:: morfessor.evaluation
   :members:


Code Examples for using library interface
=========================================

Segmenting new data using an existing model
-------------------------------------------
::

    import morfessor

    io = morfessor.MorfessorIO()

    model = io.read_binary_model_file('model.bin')

    words = ['words', 'segmenting', 'morfessor', 'unsupervised']

    for word in words:
        print(model.viterbi_segment(word))


Testing type vs token models
----------------------------
::

    import morfessor

    io = morfessor.MorfessorIO()

    train_data = list(io.read_corpus_file('training_data'))

    model_types = morfessor.BaselineModel()
    model_logtokens = morfessor.BaselineModel()
    model_tokens = morfessor.BaselineModel()

    model_types.load_data(train_data, count_modifier=lambda x: 1)
    def log_func(x):
        return int(round(math.log(x + 1, 2)))
    model_logtokens.load_data(train_data, count_modifier=log_func)
    model_tokens.load_data(train_data)

    models = [model_types, model_logtokens, model_tokens]

    for model in models:
        model.train_batch()

    goldstd_data = io.read_annotations_file('gold_std')
    ev = morfessor.MorfessorEvaluation(goldstd_data)
    results = [ev.evaluate_model(m) for m in models]

    wsr = morfessor.WilcoxonSignedRank()
    r = wsr.significance_test(results)
    WilcoxonSignedRank.print_table(r)

The equivalent of this on the command line would be: ::

    morfessor-train -s model_types -d ones training_data
    morfessor-train -s model_logtokens -d log training_data
    morfessor-train -s model_tokens training_data

    morfessor-evaluate gold_std morfessor-train morfessor-train morfessor-train


Testing different amounts of supervision data
---------------------------------------------

