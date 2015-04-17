#!/bin/sh
cd /iesl/canvas/arvind/universalSchema/factorie
if [ ! -e CP.hack ]; then
    echo "Run make_cp.sh script first"
    exit    
fi

classpath=`cat CP.hack`
app="java -Xmx40g -cp ${classpath} cc.factorie.app.nlp.embeddings.transRelations.TestTransE"
$app --train=../data//train --dev=../data//dev --test=../data//test --size $1 --rate $2 --regularizer $3 --negative $4 --threads $5 --margin $6 --iterations 10 --frequency 10 
