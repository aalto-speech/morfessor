from __future__ import division
import fileinput


def main():
    tot_morph_count = 0
    tot_length = 0

    for line in fileinput.input():
        word, segm = line.strip().split(None, 1)
        segmentations = segm.split(',')
        num_morphs = [len([x for x in s.split(None) if x.strip().strip("~") != ""]) for s in segmentations]

        tot_morph_count += sum(num_morphs) / len(num_morphs)
        tot_length += len(word)

    print(tot_length / tot_morph_count)


if __name__ == "__main__":
    main()