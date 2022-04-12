#!/bin/bash

FOLDER_NAME=Train
DATASET_PATH=`pwd`/../Datasets/$FOLDER_NAME/
DONE_PATH=`pwd`/../Datasets/${FOLDER_NAME}_Done/

test -d $DONE_PATH || mkdir $DONE_PATH
cd $DATASET_PATH || exit

lastDone=1

for i in `ls -1`;do
    echo $i
    mv $i $DONE_PATH/${lastDone}
    lastDone=$((lastDone+1))
done

rmdir $DATASET_PATH
mv $DONE_PATH $DATASET_PATH