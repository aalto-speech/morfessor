#!/usr/bin/env python

from __future__ import unicode_literals

import argparse
import collections
import morfessor
import sys

from morfessor import ArgumentException

ALIGN_LOSSES = morfessor.baseline.AlignedTokenCountCorpusWeight.align_losses

def get_argparser():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('modelfile', metavar='<modelfile>')
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
    return parser

def main(args):
    io = morfessor.io.MorfessorIO(encoding=args.encoding)

    assert args.alignref is not None
    assert args.alignseg is not None
    assert args.aligngold is not None

    updater = morfessor.baseline.AlignedTokenCountCorpusWeight(
        io._read_text_file(args.alignseg),
        io._read_text_file(args.alignref),
        0,
        linguistic_dev=io._read_text_file(args.aligngold))

    name = args.modelfile
    print('Adjusting pseudocounts for model {}'.format(name))
    sys.stdout.flush()
    model = io.read_any_model(name)
    costs, direction, tot_tokens = updater.calculate_costs(model)
    morph_totals = updater.morph_totals
    morph_scores = updater.morph_scores

    ranked_morphs = []
    for (morph, score) in morph_scores.items():
        total = morph_totals[morph]
        ranked_morphs.append((abs(float(score))/total, score, total, morph))
    ranked_morphs.sort(reverse=True)
    print('Top morphs:')
    for tpl in ranked_morphs[:20]:
        print('rel {}\tscore {}\ttotal {}\tmorph "{}"'.format(*tpl))


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
