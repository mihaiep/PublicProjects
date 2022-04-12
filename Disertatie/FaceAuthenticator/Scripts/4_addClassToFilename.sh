#!/bin/bash

DIR=$1

cd ../Datasets/$DIR
for i in `ls -1`;do
    cd $i
    pwd
    for name in *; do
        mv "$name" "${i}_$name"
    done
    cd ..
done