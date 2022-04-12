#!/bin/bash

splitProc=85
DATASET_PATH=`pwd`/../Datasets/Train
TEST_PATH=`pwd`/../Datasets/Test

test -d $TEST_PATH || mkdir $TEST_PATH
cd $DATASET_PATH || exit

for i in `ls`;do
	echo $i
    test -d $TEST_PATH/$i || mkdir $TEST_PATH/$i
    imagesNum=`ls -1 $i | wc -l`
    keepNum=`echo $((imagesNum*splitProc/100))`

    lst=(`ls $i| shuf`)
    cnt=$((imagesNum-keepNum))
    for((k=1;k<=cnt;k++));do
        mv $i/${lst[k]} $TEST_PATH/$i
    done
done
