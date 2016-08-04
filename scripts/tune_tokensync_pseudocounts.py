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
    add_arg('modelfile', metavar='<modelfile>',
            help='Restricted Morfessor model')
    add_arg('infile', metavar='<infile>',
            help='Segmentation lexicon of the format: '
                 '<float pseudocount> TAB <word> TAB <space separated morphs>')
    add_arg('outfile', metavar='<outfile>',
            help='Segmentation lexicon of the format: '
                 '<float pseudocount> TAB <word> TAB <space separated morphs>')
    add_arg('roundedoutfile', metavar='<roundedoutfile>',
            help='Segmentation lexicon of the format: '
                 '<int pseudocount> TAB <word> TAB <space separated morphs>')
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
    add_arg('--learning-rate', dest='rate', type=float, default=0.5,
            help='Scale adjustments by this factor.')
    add_arg('--smooth', dest='smooth', type=float, default=100.,
            help='Additive smoothing of estimates: '
                 'add this number of neutral occurrences to each morph.')
    return parser

def main(args):
    io = morfessor.io.MorfessorIO(encoding=args.encoding)

    assert args.alignref is not None
    assert args.alignseg is not None
    assert args.aligngold is not None

    updater = morfessor.AlignedTokenCountCorpusWeight(
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
    morph_scores_pos = updater.morph_scores_pos
    morph_scores_neg = updater.morph_scores_neg
    rel_morph_scores = collections.Counter()

    ling_morphs = set()
    for line in io._read_text_file(args.infile):
        line = line.strip()
        try:
            count, word, morphstr = line.split('\t')
            morphs = morphstr.split()
            ling_morphs.update(morphs)
        except ValueError:
            print('cant parse line {}'.format(line))
            raise

    ranked_morphs = []
    for (morph, total) in morph_totals.items():
        if morph not in ling_morphs:
            print('skipping unsegmentable {} ({})'.format(morph, total))
            continue
        pos = float(max(0, morph_scores_pos[morph]))
        neg = float(max(0, morph_scores_neg[morph]))
        score = (pos - neg) / (total + args.smooth)
        rel_morph_scores[morph] = score
        ranked_morphs.append((abs(score),
                             pos, morph_scores_pos[morph],
                             neg, morph_scores_neg[morph],
                             total, morph))
    ranked_morphs.sort(reverse=True)
    print('Top morphs:')
    for tpl in ranked_morphs: # [:20]:
        print('absrel {}\tpos {} ({})\tneg {} ({})\ttotal {}\tmorph "{}"'.format(*tpl))

    # file format: current count (float), word, linguistic morphs
    #   created by initializer, which optionally applies log and/or multiplier to true counts
    #   both input and output (with current count updated)
    # also output: current count (rounded), word

    total_in = 0.
    total_out = 0.
    with io._open_text_file_write(args.outfile) as outfobj:
        with io._open_text_file_write(args.roundedoutfile) as roundfobj:
            for line in io._read_text_file(args.infile):
                line = line.strip()
                try:
                    count, word, morphstr = line.split('\t')
                    count = float(count)
                    total_in += count
                    morphs = morphstr.split()
                    scores = [rel_morph_scores[morph] for morph in morphs]
                    score = sum(scores) / len(scores)
                    # positive score: oversegmented,  needs higher corpus weight
                    # negative score: undersegmented, needs lower corpus weight
                    count *= (1. + (score * args.rate))
                    total_out += count
                    outfobj.write('{}\t{}\t{}\n'.format(count, word, morphstr))

                    rounded = max(1, int(round(count)))
                    roundfobj.write('{}\t{}\n'.format(rounded, word))
                except ValueError:
                    print('cant parse line {}'.format(line))
                    raise
    print('total in: {}, total out: {}, ratio: {}'.format(
        total_in, total_out, total_out / total_in))


if __name__ == "__main__":
    parser = get_argparser()
    try:
        args = parser.parse_args(sys.argv[1:])
        main(args)
    except ArgumentException as e:
        parser.error(e)
    except Exception as e:
        raise
