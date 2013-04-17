#!/usr/bin/env python

from distribute_setup import use_setuptools
use_setuptools()

from setuptools import setup

import re
main_py = open('morfessor/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__ = '([^']+)'", main_py))

requires = [
    #    'progressbar',
]

setup(name='Morfessor',
      version=metadata['version'],
      author=metadata['author'],
      author_email='morfessor@cis.hut.fi',
      url='http://www.cis.hut.fi/projects/morpho/',
      description='Morfessor',
      py_modules=['morfessor', 'distribute_setup'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
      ],
      license="BSD",
      scripts=['scripts/morfessor', 'scripts/morfessor-train',
               'scripts/morfessor-segment'],
      install_requires=requires,
      )
