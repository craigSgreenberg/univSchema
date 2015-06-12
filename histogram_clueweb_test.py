
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

def histogram_clueweb_test(filepath, clueweb_counts, sep='\t', is_cum=True):
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
    if is_cum:
        items = [(cnt, cnt_of_cnts) for cnt, cnt_of_cnts in cntr.iteritems()]
        items.sort()
        cur_cnt = sum([cnt_of_cnts for _,cnt_of_cnts in items])
        for cnt, cnt_of_cnts in items:
            print cur_cnt, "have", cnt, "or more occurances."
            cur_cnt -= cnt_of_cnts
    else:
        for cnt, cnt_of_cnts in cntr.most_common():
            print cnt, cnt_of_cnts

def main():
    print 'loading clueweb'
    clueweb_filepath = '/iesl/canvas/proj/processedClueweb12/clueweb/clueweb_entity_types.v2.tsv'
    clueweb_counts = load_clueweb_counts(clueweb_filepath)
    v = [(v,k) for k,v in clueweb_counts.iteritems()]
    v.sort(reverse=True)
    print "The thousand freebaseids with the most frequent occurance in clueweb:"
    print v[:1000]
    print 'clueweb loaded'
    print 'loading freebase'
    # switching from MSR data to custom data
    #freebase_test_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseTest70.v0.tsv'
    freebase_test_filepath = '/iesl/canvas/proj/processedClueweb12/freebase/freebaseTest.v2.tsv'
    histogram_clueweb_test(freebase_test_filepath, clueweb_counts)

if __name__ == '__main__':
    main()
