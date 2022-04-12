#!/bin/bash

DIR=$1
outFile=`pwd`/../Datasets/Utils/ClassesCount_${DIR}.txt
test -d ../Datasets/Utils || mkdir "../Datasets/Utils"

echo $outFile
cd ../Datasets/$DIR
for i in `ls -1`; do
    number=`ls -1 $i | wc -l`
    echo "$i,$number" >> "$outFile"
done
