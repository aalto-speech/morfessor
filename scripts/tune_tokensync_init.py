#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import morfessor
import sys
import re
import math

# Markers used for indicating intra-word boundaries
RE_MARKERS = re.compile(r'[+@]')

from morfessor import ArgumentException

ALIGN_LOSSES = morfessor.AlignedTokenCountCorpusWeight.align_losses

def get_argparser():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('infile', metavar='<infile>',
            help='Segmentation lexicon of the format: '
                 '<true count> TAB <word> TAB <space separated morphs>')
    add_arg('outfile', metavar='<outfile>',
            help='Segmentation lexicon of the format: '
                 '<float pseudocount> TAB <word> TAB <space separated morphs>')
    add_arg('roundedoutfile', metavar='<roundedoutfile>',
            help='Segmentation lexicon of the format: '
                 '<int pseudocount> TAB <word> TAB <space separated morphs>')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')
    add_arg('-d', '--dampening', dest="dampening", type=str, default='log',
            metavar='<type>', choices=['none', 'log', 'ones'],
            help="frequency dampening for training data ('none', 'log', or "
                 "'ones'; default '%(default)s')")
    add_arg('-m', '--multiplier', dest='multiplier', type=float, default=100.,
            metavar='<float>',
            help="multiply dampened frequency before rounding. "
                 "Affects the granularity available for adjustments. "
                 "Must be compensated for in corpus weight.")

    return parser

def main(args):
    io = morfessor.io.MorfessorIO(encoding=args.encoding)

    # Set frequency dampening function
    if args.dampening == 'none':
        dampfunc = lambda x: x
    elif args.dampening == 'log':
        dampfunc = lambda x: math.log(x + 1, 2)
    elif args.dampening == 'ones':
        dampfunc = lambda x: 1
    else:
        raise ArgumentException("unknown dampening type '%s'" % args.dampening)

    with io._open_text_file_write(args.outfile) as outfobj:
        with io._open_text_file_write(args.roundedoutfile) as roundfobj:
            for line in io._read_text_file(args.infile):
                line = line.strip()
                try:
                    count, word, morphstr = line.split('\t')
                    morphstr = RE_MARKERS.sub('', morphstr)
                    count = dampfunc(float(count)) * args.multiplier
                    outfobj.write('{}\t{}\t{}\n'.format(count, word, morphstr))

                    rounded = max(1, int(round(count)))
                    roundfobj.write('{}\t{}\n'.format(rounded, word))
                except ValueError:
                    print('cant parse line {}'.format(line))
                    raise


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
