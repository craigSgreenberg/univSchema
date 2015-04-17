#!/bin/sh
cd /iesl/canvas/arvind/universalSchema/factorie
if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit    
fi

classpath=`cat CP.hack`
app="java -Xmx${7}g -cp ${classpath} cc.factorie.app.nlp.embeddings.UniversalSchema"
$app --train=../data//train --dev=../data//dev --test=../data//test --size $1 --rate $2 --regularizer $3 --negative $4 --threads $5 --options $6  --epochs 30 --frequency 10 --writeOutput false 
