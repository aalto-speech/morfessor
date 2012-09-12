#!/usr/bin/env python

from distutils.core import setup

setup(name='Morfessor',
      version='1.99',
      author=','.join(open('AUTHORS').readlines()),
      author_email='sami.virpioja@aalto.fi',
      maintainer='Peter Smit',
      maintainer_email='peter.smit@aalto.fi',
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
