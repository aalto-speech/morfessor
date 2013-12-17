General
=======

.. _morfessor-tech-report:

Morfessor 2.0 Technical Report
------------------------------

The work done in Morfessor 2.0 is described in detail in the Morfessor 2.0
Technical Report [TechRep]_. The report is available for download from
http://urn.fi/URN:ISBN:978-952-60-5501-5.


Terminology
-----------

Unlike previous Morfessor implementations, Morfessor 2.0 is, in
principle, applicable to any string segmentation task. Thus we use
terms that are not specific to morphological segmentation task.

The task of the algorithm is to find a set of *constructions* that
describe the provided training corpus efficiently and accurately. The
training corpus contains a collection of *compounds*, which are the
largest sequences that a single construction can hold. The smallest
pieces of constructions and compounds are called *atoms*.

For example, in morphological segmentation, compounds are word forms,
constructions are morphs, and atoms are characters. In chunking,
compounds are sentences, constructions are phrases, and atoms are
words.

Citing
------

The authors do kindly ask that you cite the Morfessor 2.0 techical report
 [TechRep]_ when using this tool in academic publications.

In addition, when you refer to the Morfessor algorithms, you should cite the
respective publications where they have been introduced. For example, the first
Morfessor algorithm was published in [Creutz2002]_ and the semi-supervised
extension in [Kohonen2010]_. See [TechRep]_ for further information on the
relevant publications.

.. [TechRep] Sami Virpioja, Peter Smit, Stig-Arne Grönroos, and Mikko Kurimo. Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline. Aalto University publication series SCIENCE + TECHNOLOGY, 25/2013. Aalto University, Helsinki, 2013. ISBN 978-952-60-5501-5.

.. [Creutz2002] Mathias Creutz and Krista Lagus. Unsupervised discovery of morphemes. In Proceedings of the Workshop on Morphological and Phonological Learning of ACL-02, pages 21-30, Philadelphia, Pennsylvania, 11 July, 2002. 

.. [Kohonen2010] Oskar Kohonen, Sami Virpioja and Krista Lagus. Semi-supervised learning of concatenative morphology. In Proceedings of the 11th Meeting of the ACL Special Interest Group on Computational Morphology and Phonology, pages 78-86, Uppsala, Sweden, July 2010. Association for Computational Linguistics.

