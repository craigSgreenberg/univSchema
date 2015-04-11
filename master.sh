#!/bin/sh
#dimList="10 50 100"
#lrList="1.0 0.1 0.01"
#regList="0.1 0.01 0.001"
#negativeList="1 10 20"
dimList="100"
lrList="0.1"
regList="0.001"
negativeList="1"
threadList="20"
for dim in $dimList
do
for lr in $lrList
do
for reg in $regList
do
for neg in $negativeList
do
for thread in $threadList
do
qsub -S /bin/sh -l mem_token=30G -l mem_free=30G run.sh $dim $lr $reg $neg $thread
#./run.sh $dim $lr $reg $neg $thread
done
done
done
done
done
