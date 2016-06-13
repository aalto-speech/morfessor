#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import morfessor
import sys

from morfessor import ArgumentException

def get_argparser():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('modelfiles',
        metavar='<modelfile>',
        nargs='+')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    add_arg('--aligned-reference', dest='alignref', default=None,
            metavar='<file>',
            help='FIXME')
    add_arg('--aligned-to-segment', dest='alignseg', default=None,
            metavar='<file>',
            help='FIXME')
    add_arg('--aligned-loss', dest="alignloss", type=str, default='abs',
            metavar='<type>', choices=['abs', 'square', 'zeroone', 'tot'],
            help="loss function for FIXME ('abs', 'square', 'zeroone' or"
                 "'tot'; default '%(default)s')")
    return parser

def get_aligned_token_cost(updater, model):
    #distribution = OnlyDiffDistr()
    #distribution = ScatterDistr()
    (cost, direction) = updater.evaluation(
        model, distribution=None)
    return (cost, direction)

def main(args):
    io = morfessor.io.MorfessorIO(encoding=args.encoding)

    assert args.alignref is not None
    assert args.alignseg is not None

    postfunc = None
    if args.alignloss == 'abs':
        lossfunc = abs
    elif args.alignloss == 'square':
        lossfunc = lambda x: x**2
    elif args.alignloss == 'zeroone':
        lossfunc = lambda x: 0 if x == 0 else 1
    elif args.alignloss == 'tot':
        lossfunc = lambda x: x
        postfunc = abs
    else:
        raise ArgumentException(
            "unknown alignloss type '{}'".format(args.alignloss))
    updater = morfessor.baseline.AlignedTokenCountCorpusWeight(
        io._read_text_file(args.alignseg),
        io._read_text_file(args.alignref),
        0,
        lossfunc,
        postfunc)

    for name in args.modelfiles:
        print('Evaluating model {}'.format(name))
        sys.stdout.flush()
        model = io.read_any_model(name)
        cost, direction = get_aligned_token_cost(updater, model)
        print('{}\t{}\t{}'.format(cost, direction, name))
        sys.stdout.flush()


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
