__author__ = 'pat'
from random import shuffle, randint
from collections import defaultdict
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Filter and split data for universal schema experiments.')
    parser.add_argument('-t', '--textInput', help='Input file name of relations from text', required=True)
    parser.add_argument('-kb', '--kbInput', help='Input file name of relations from knowledge base.', required=True)
    parser.add_argument('-o', '--output', help='Path to output location', default='', required=False)
    parser.add_argument('-pre', '--prefix', help='prefix to append to output files', default='', required=False)
    parser.add_argument('-tsv', '--parseTsv', help='input data is in tsv format', default=False, dest='parseTsv',
                        action='store_true', required=False)
    parser.add_argument('-min', '--minRelationCount', help='Filter relations occurring less than min times',
                        default=10, type=int, required=False)
    parser.add_argument('-neg', '--negSamples', help='Number of negative samples per example.', default=10, type=int,
                        required=False)
    return parser.parse_args()


# parse data in arvind format [e1,e2 \t rel \t label]
def parse_arvind(line):
    parts = line.split('\t')
    pair = parts[0]
    rel = parts[1]
    return pair, rel


# parse data in tsv format
def parse_tsv(line):
    parts = line.split('\t')
    pair = parts[0] + ',' + parts[1]
    rel = parts[2].replace(' ', ',')
    return pair, rel


def run():
    args = parse_args()
    text_in_file = args.textInput
    kb_in_file = args.kbInput
    out_path = args.output + args.prefix
    min_count = args.minRelationCount
    tsv = args.parseTsv
    neg_samples = args.negSamples
    out_names = [out_path + "train", out_path + "dev", out_path + "test"]
    splits = [[], [], []]
    pair_set = set()
    relation_set = set()

    # save all pairs first to negative sample
    relation_pairs = defaultdict(list)
    for line in open(text_in_file, 'r'):
        if line.strip():
            pair, rel = parse_tsv(line) if tsv else parse_arvind(line)
            relation_pairs[rel].append(pair)

    filtered_rel_pairs = [(pair, rel) for rel, pair_list in relation_pairs.iteritems() for pair in pair_list if
                          len(pair_list) > min_count]

    # export all data from text relations to train
    shuffle(filtered_rel_pairs)
    for pair, relation in filtered_rel_pairs:
        # enforce that train set contains every pair and every relation
        if pair not in pair_set:
            pair_set.add(pair)
        if relation not in relation_set:
            relation_set.add(relation)
        splits[0].append(pair + '\t' + relation + '\t' + '1\n')

    filtered_pairs = [pair for pair, rel in filtered_rel_pairs]
    filtered_pair_set = set(filtered_pairs)

    for line in open(kb_in_file, 'r'):
        pair, relation = parse_tsv(line) if tsv else parse_arvind(line)
        if pair in filtered_pair_set:
            # choose which split to add this example to
            index = 0 if pair not in pair_set or relation not in relation_set else randint(0, 2)
            splits[index].append(pair + '\t' + relation + '\t' + '1\n')
            # add negative samples to dev and test
            if index > 0:
                for i in range(1, neg_samples):
                    negative_pair = filtered_pairs[randint(0, len(filtered_pairs) - 1)]
                    splits[randint(1, 2)].append(negative_pair + "\t" + relation + "\t0\n")

            # enforce that train set contains every pair and every relation
            if pair not in pair_set:
                pair_set.add(pair)
            if relation not in relation_set:
                relation_set.add(relation)

    print(len(splits[0]), len(splits[1]), len(splits[2]))

    for (split, name) in zip(splits, out_names):
        shuffle(split)
        f = open(name, 'w')
        for line in split:
            f.write(line)
        f.close()


if __name__ == "__main__":
    run()