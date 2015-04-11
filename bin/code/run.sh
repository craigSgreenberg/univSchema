#!/bin/sh
cd /iesl/canvas/arvind/universalSchema/code
if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit    
fi

classpath=`cat CP.hack`
app="java -Xmx30g -cp ${classpath} cc.factorie.app.nlp.embeddings.UniversalSchema"
$app --train=../data/synthesisProject//train --dev=../data/synthesisProject//dev --test=../data/synthesisProject//test --size $1 --rate $2 --regularizer $3 --negative $4 --threads $5 --options 2  --epochs 100 --frequency 100 
