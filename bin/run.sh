#!/bin/sh
cd /iesl/canvas/arvind/universalSchema/old/
if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit    
fi

classpath=`cat CP.hack`
app="java -Xmx50g -cp ${classpath} cc.factorie.app.nlp.embeddings.WordVec"
$app --train=../data///train --dev=../data//dev --test=../data//test --size $1 --rate $2 --regularizer $3 --negative $4 --threads $5 --options 2  --epochs 50 --frequency 10 --writeOutput false 
