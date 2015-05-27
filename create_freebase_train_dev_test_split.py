
from collections import defaultdict

import random
random.seed(42)

def get_type2freebaseid(filepath, sep='\t'):
    type2freebaseid = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            #/m/010bvf       %%base%%type_ontology%%inanimate        1
            freebaseid, ftype, val = line.split(sep)
            assert val == '1'
            type2freebaseid[ftype].append(freebaseid)
    return type2freebaseid

def split_train_dev_test(type2freebaseid, selected_types, train_per=.6, dev_per=.1):
    train, dev, test = [], [], []
    for ftype, freebaseids in type2freebaseid.iteritems():
        if not ftype in selected_types:
            train.extend([(ftype, fid) for fid in freebaseids])
        else:
            random.shuffle(freebaseids)
            num_train = int(len(freebaseids) * train_per)
            num_dev = int(len(freebaseids) * dev_per)
            trainids = freebaseids[:num_train]
            devids = freebaseids[num_train:num_train+num_dev]
            testids = freebaseids[num_train+num_dev:]
            train.extend([(ftype, fid) for fid in trainids])
            dev.extend([(ftype, fid) for fid in devids])
            test.extend([(ftype, fid) for fid in testids])
    return train,dev,test

def write_to_file(ftype_fid, filepath, sep='\t'):
    with open(filepath, 'w') as f:
        for ftype, fid in ftype_fid:
            f.write('{fid}{sep}{ftype}{sep}1\n'.format(fid=fid, ftype=ftype, sep=sep))

def main():
    # choose m of the n most common types
    n = 1000
    m = 100
    freebase_type_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseAll.v0.tsv'
    train_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseTrain.v1.tsv'
    dev_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseDev.v1.tsv'
    test_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseTest.v1.tsv'
    type2freebaseid = get_type2freebaseid(freebase_type_filepath)
    counts = [(len(v), k) for k,v in type2freebaseid.iteritems()]
    counts.sort(reverse=True)
    selected_types = random.sample([t for (_, t) in counts[:n]],m)
    train, dev, test = split_train_dev_test(type2freebaseid, selected_types)
    write_to_file(train, train_filepath)
    write_to_file(dev, dev_filepath)
    write_to_file(test, test_filepath)

if __name__ == '__main__':
    main()
