#!/usr/bin/env python
"""
Morfessor 2.0 - Python implementation of the Morfessor method
"""
import logging
import sys
import types

from .baseline import BaselineModel
from .cmd import main, get_default_argparser
from .exception import MorfessorException, ArgumentException
from .io import MorfessorIO

__all__ = ['MorfessorException', 'ArgumentException', 'MorfessorIO',
           'BaselineModel', 'main', 'get_default_argparser']

__version__ = '2.0.0alpha3'
__author__ = 'Sami Virpioja, Peter Smit'
__author_email__ = "morfessor@cis.hut.fi"
show_progress_bar = True

_logger = logging.getLogger(__name__)


def get_version():
    return __version__


def _progress(iter_func):
    """Decorator/function for displaying a progress bar when iterating
    through a list.

    iter_func can be both a function providing a iterator (for decorator
    style use) or an iterator itself.

    No progressbar is displayed when the show_progress_bar variable is set to
     false.

    If the progressbar module is available a fancy percentage style
    progressbar is displayed. Otherwise 60 dots are printed as indicator.

    """

    if not show_progress_bar:
        return iter_func

    #Try to see or the progressbar module is available, else fabricate our own
    try:
        from progressbar import ProgressBar
    except ImportError:
        class SimpleProgressBar:
            """Create a simple progress bar that prints 60 dots on a single
            line, proportional to the progress """
            NUM_DOTS = 60

            def __call__(self, it):
                self.it = iter(it)
                self.i = 0

                # Dot frequency is determined as ceil(len(it) / NUM_DOTS)
                self.dotfreq = (len(it) + self.NUM_DOTS - 1) // self.NUM_DOTS
                if self.dotfreq < 1:
                    self.dotfreq = 1

                return self

            def __iter__(self):
                return self

            def __next__(self):
                self.i += 1
                if self.i % self.dotfreq == 0:
                    sys.stderr.write('.')
                    sys.stderr.flush()
                try:
                    return next(self.it)
                except StopIteration:
                    sys.stderr.write('\n')
                    raise

            #Needed to be compatible with both Python2 and 3
            next = __next__

        ProgressBar = SimpleProgressBar

    # In case of a decorator (argument is a function),
    # wrap the functions result in a ProgressBar and return the new function
    if isinstance(iter_func, types.FunctionType):
        def i(*args, **kwargs):
            if logging.getLogger(__name__).isEnabledFor(logging.INFO):
                return ProgressBar()(iter_func(*args, **kwargs))
            else:
                return iter_func(*args, **kwargs)
        return i

    #In case of an iterator, wrap it in a ProgressBar and return it.
    elif hasattr(iter_func, '__iter__'):
        return ProgressBar()(iter_func)

    #If all else fails, just return the original.
    return iter_func
