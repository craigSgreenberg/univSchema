
from collections import defaultdict

import random
random.seed(42)

def get_type2freebaseid(filepath, sep='\t', type_filter=lambda x: True):
    type2freebaseid = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            #/m/010bvf       %%base%%type_ontology%%inanimate        1
            freebaseid, ftype, val = line.split(sep)
            assert val == '1'
            if type_filter(ftype):
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

def reformat_pat_and_arvinds_entity_file(infilepath='/iesl/canvas/proj/processedClueweb12/freebase/iesl/entity_to_fbtypes',
                                         outfilepath='/iesl/canvas/proj/processedClueweb12/freebase/iesl/entity_to_fbtypes.tsv'):
    with open(infilepath) as inf:
        with open(outfilepath, 'w') as outf:
            for line in inf:
                line = line.strip()
                if not line:
                    continue
                entity, types = line.split('\t')
                for t in types.split(','):
                    outf.write('%s\t%s\t1\n'%(entity, t))

def write_train_dev_test_splits():
    """
    Choose m of the n most common types for tests (the rest go into train only),
    split into train/dev/test."""
    # Set filepaths
    # replaced MSR file with Pat and Arvind's entity file...
    # freebase_type_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseAll.v0.tsv'
    freebase_type_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/iesl/entity_to_fbtypes.tsv'
    train_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseTrain.v2.tsv'
    dev_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseDev.v2.tsv'
    test_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseTest.v2.tsv'
    # Set parameters
    n = 1000
    m = 100
    # Select test types
    type_filter = lambda x: x[:3] != "/m/" and x[:10] != "/freebase/"
    type2freebaseid = get_type2freebaseid(freebase_type_filepath, type_filter=type_filter)
    counts = [(len(v), k) for k,v in type2freebaseid.iteritems()]
    counts.sort(reverse=True)
    selected_types = random.sample([t for (_, t) in counts[:n]],m)
    # Split into train/dev/test
    train, dev, test = split_train_dev_test(type2freebaseid, selected_types)
    # Write splits to files
    write_to_file(train, train_filepath)
    write_to_file(dev, dev_filepath)
    write_to_file(test, test_filepath)

def main():
    #reformat_pat_and_arvinds_entity_file()
    write_train_dev_test_splits()

if __name__ == '__main__':
    main()
