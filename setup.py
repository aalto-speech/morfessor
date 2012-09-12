#!/usr/bin/env python

from distutils.core import setup

import re
main_py = open('morfessor.py').read()
metadata = dict(re.findall("__([a-z]+)__ = '([^']+)'", main_py))

setup(name='Morfessor',
      version=metadata['version'],
      author=metadata['author'],
      author_email=metadata['author_email'],
      url='http://www.cis.hut.fi/projects/morpho/',
      description='Morfessor',
      packages=['morfessor'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
      ],
      license="BSD",
     )
