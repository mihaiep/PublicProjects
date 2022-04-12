#!/bin/sh

rootPath='../Datasets/Train/'

while read line;do
    fileName=`echo $line | awk '{print $1}'`
    filePath=$rootPath$fileName
    if [ -f $filePath ]; then
        fileID=`echo $line | awk '{print $2}' | sed 's/\r//'`
        dirName=$rootPath$fileID
        echo "Filename: $fileName - FileID: $fileID - Path: $filePath -> moving to: $dirName"
        test -d "$dirName" || mkdir $dirName
        mv $filePath $dirName
    else
        echo "Not found: $fileName"
    fi
done < identity_CelebA.txt