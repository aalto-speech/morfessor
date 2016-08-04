#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import morfessor
import sys

from morfessor import ArgumentException

ALIGN_LOSSES = morfessor.AlignedTokenCountCorpusWeight.align_losses

def get_argparser():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('modelfiles',
        metavar='<modelfile>',
        nargs='+',
        help='Restricted Morfessor models to evaluate')
    add_arg('-e', '--encoding', dest='encoding', metavar='<encoding>',
            help='Encoding of input and output files (if none is given, '
                 'both the local encoding and UTF-8 are tried).')

    add_arg('--aligned-reference', dest='alignref', default=None,
            metavar='<file>',
            help='Reference corpus '
                 '(the side of the parallel corpus with low '
                 'morphological complexity)')
    add_arg('--aligned-to-segment', dest='alignseg', default=None,
            metavar='<file>',
            help='Corpus to segment, as unsegmented tokens '
                 '(the side of the parallel corpus with high '
                 'morphological complexity)')
    add_arg('--aligned-linguistic', dest='aligngold', default=None,
            metavar='<file>',
            help='Corpus to segment, as linguistic gold standard tokens '
                 '(Same text as --aligned-to-segment). Optional.')
    add_arg('--aligned-loss', dest="alignloss", type=str, default='abs',
            metavar='<type>',
            choices=ALIGN_LOSSES,
            help="loss function for token count tuning "
                 "('abs', 'square', 'zeroone' or"
                 "'tot'; default '%(default)s')")
    return parser

def main(args):
    io = morfessor.io.MorfessorIO(encoding=args.encoding)

    assert args.alignref is not None
    assert args.alignseg is not None

    if args.alignloss not in ALIGN_LOSSES:
        raise ArgumentException(
            "unknown alignloss type '{}'".format(args.alignloss))
    if args.aligngold is not None:
        aligngold = io._read_text_file(args.aligngold)
    else:
        aligngold = None
    updater = morfessor.AlignedTokenCountCorpusWeight(
        io._read_text_file(args.alignseg),
        io._read_text_file(args.alignref),
        0,
        loss=args.alignloss,
        linguistic_dev=aligngold)

    best_costs = {label: None for label in ALIGN_LOSSES}
    best_models = {}
    for name in args.modelfiles:
        print('Evaluating model {}'.format(name))
        sys.stdout.flush()
        model = io.read_any_model(name)
        costs, direction, tot_tokens = updater.calculate_costs(model)
        for (label, cost) in zip(ALIGN_LOSSES, costs):
            print('{}\t{}\t{}\t{}'.format(label, cost, direction, name))
            if best_costs[label] is None or best_costs[label] > cost:
                best_costs[label] = cost
                best_models[label] = name
        sys.stdout.flush()
    for label in ALIGN_LOSSES:
        selected = '(selected)' if label == args.alignloss else '\t'
        print('best model for loss {}: {}\t{} {}'.format(
            label, selected, best_models[label], best_costs[label]))


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
