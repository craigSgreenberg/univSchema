
from collections import defaultdict

import random
random.seed(42)

def get_type2freebaseid(filepath, sep='\t'):
    type2freebaseid = defaultdict()
    i = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            #/m/010bvf       %%base%%type_ontology%%inanimate        1
            freebaseid, ftype, val = line.split(sep)
            assert val == '1'
            type2freebaseid[ftype] = freebaseid
            i += 1
            if i > 500000:
                break
    return type2freebaseid

def split_train_dev_test(type2freebaseid, selected_types, train_per=.60, dev_per=.1, test_per=.3):
    train, dev, test = [], [], []
    for ftype, freebaseids in type2freebaseid:
        if not ftype in selected_types:
            train.extend([(ftype, fid) for fid in freebaseids])
        else:
            random.shuffle(freebaseids)
            num_train = int(len(freebaseids)*train_per)
            num_dev = int(len(freebaseids)*dev_per)
            trainids = freebaseids[:num_train]
            devids = freebaseids[num_train:num_train+num_dev]
            testids = freebaseids[num_train+num_dev:]
            train.extend([(ftype, fid) for fid in trainids])
            dev.extend([(ftype, fid) for fid in devids])
            test.extend([(ftype, fid) for fid in testids])
    return train,dev,test

def main():
    # choose m of the n most common types
    n = 1000
    m = 100
    freebase_type_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseAll.v0.tsv'
    train_filepath = ''
    dev_filepath = ''
    test_filepath = ''
    type2freebaseid = get_type2freebaseid(freebase_type_filepath)
    counts = [(len(v), k) for k,v in type2freebaseid.iteritems()]
    counts.sort(reverse=True)
    selected_types = random.sample([t for (_, t) in counts[:n]],m)


if __name__ == '__main__':
    main()
