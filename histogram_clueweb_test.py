
from collections import defaultdict
from collections import Counter

def load_clueweb_counts(filepath, sep='\t'):
    d = defaultdict(int)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            freebaseid = line.split(sep)[0]
            d[freebaseid] += 1
    return d

def histogram_clueweb_test(filepath, clueweb_counts, sep='\t'):
    d = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            freebaseid = line.split(sep)[0]
            d[freebaseid] = clueweb_counts[freebaseid]
    print 'freebase loaded'
    print 'starting counting'
    cntr = Counter(d.values())
    print 'counting complete'
    #keys = list(cntr.elements())
    #keys.sort()
    for cnt, cnt_of_cnts in cntr.most_common():
        print cnt, cnt_of_cnts

def main():
    print 'loading clueweb'
    clueweb_filepath = '/iesl/canvas/proj/processedClueweb12/clueweb/clueweb_entity_types.v2.tsv'
    clueweb_counts = load_clueweb_counts(clueweb_filepath)
    print clueweb_counts.values().sort()
    print 'clueweb loaded'
    print 'loading freebase'
    freebase_test_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseTest70.v0.tsv'
    histogram_clueweb_test(freebase_test_filepath, clueweb_counts)

if __name__ == '__main__':
    main()
