#!/bin/sh
cd /iesl/canvas/arvind/universalSchema/factorie
if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit    
fi

classpath=`cat CP.hack`
app="java -Xmx30g -cp ${classpath} cc.factorie.app.nlp.embeddings.UniversalSchema"
$app --train=/iesl/canvas/proj/processedClueweb12/clueweb_and_msr_freebase_train_types.v1.tsv --dev=/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseDev70.v0.tsv --test=/iesl/canvas/proj/processedClueweb12/freebase/msr/msrFreebaseTest70.v0.tsv --size $1 --rate $2 --regularizer $3 --negative $4 --threads $5 --options 2  --epochs 100 --frequency 20 --tree $6
