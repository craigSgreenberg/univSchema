
def get_vocab_map(infilepath, sep='\t'):
    vocab = {}
    with open(infilepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, wordid = line.split(sep)
            vocab[wordid] = word
    return vocab

def reformat_freebase_features(filepath, vocab_map, do_something, sep='\t', featsep=':'):
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                freebaseid, features = line.split(sep)
            except ValueError:
                assert len(line.split(sep))
                print 'skipping', line
                continue
            for feature in features.split():
                wordid, tfidf = feature.split(featsep)
                word = vocab_map[wordid]
                do_something(freebaseid, word, tfidf)

def print_freebase_features(filepath, vocab_map):
    def do_print(freebaseid, word, tfidf):
        print word, tfidf
    reformat_freebase_features(filepath, vocab_map, do_print)

def write_freebase_features(filepath, vocab_map, outfilepath, sep='\t'):
    with open(outfilepath, 'w') as f:
        def do_write(freebaseid, word, tfidf):
            f.write("{freebaseid}{sep}{word}{sep}{tfidf}\n".format(freebaseid=freebaseid, word=word, tfidf=tfidf, sep=sep))
        reformat_freebase_features(filepath, vocab_map, do_write)

def main():
    infilepath = '/Users/csgreenberg/workspace/data/iesl/clueweb2012/freebase_features/unarchived/firstParaFeatures'
    vocab_map_filepath = '/Users/csgreenberg/workspace/data/iesl/clueweb2012/freebase_features/unarchived/featureMap'
    vocab_map = get_vocab_map(vocab_map_filepath)
    #print_freebase_features(infilepath, vocab_map)
    outfilepath = '/Users/csgreenberg/workspace/data/iesl/clueweb2012/freebase_features/unarchived/freebase_word_features.tsv'
    write_freebase_features(infilepath, vocab_map, outfilepath)


if __name__ == '__main__':
    main()
