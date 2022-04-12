#!/bin/bash
! test -d '../Datasets/Train/' && echo "Dir not found." && exit
cd ../Datasets/Train/
for dir in `ls -1`;do
	filesNum=$(ls $dir | wc -l)
	if ((filesNum<$1)); then
		echo "Dir $dir has under $1 image."
		rm -rf $dir
	fi
done
