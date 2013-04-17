#!/usr/bin/env python
"""
Morfessor 2.0 - Python implementation of the Morfessor method
"""
from .baseline import BaselineModel
from morfessor.io import MorfessorIO

__all__ = ['MorfessorException', 'MorfessorIO', 'BaselineModel',
           'AnnotationsModelUpdate', 'Encoding', 'CorpusEncoding',
           'AnnotatedCorpusEncoding', 'LexiconEncoding']

__version__ = '2.0.0alpha3'
__author__ = 'Sami Virpioja, Peter Smit'
__author_email__ = "morfessor@cis.hut.fi"

import logging
import sys

PY3 = sys.version_info.major == 3

try:
    # In Python2 import cPickle for better performance
    import cPickle as pickle
except ImportError:
    import pickle

_logger = logging.getLogger(__name__)


class MorfessorException(Exception):
    """Base class for exceptions in this module."""
    pass