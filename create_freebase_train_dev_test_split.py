
from __future__ import division

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
    print counts
    counts.sort(reverse=True)
    selected_types = random.sample([t for (_, t) in counts[:n]],m)
    print selected_types

if __name__ == '__main__':
    main()
