#!/bin/bash

#A script to zip files in the directory

dirstozip=${SCRATCH}/
# types of the pond compilation types, check generator
tpes=(expansion-0 expasion-1 expansin-2 expansion-3 lazy-0 lazy-1 lazy-2 lazy-3)
#dont change, adds file in the directory
adds=()
#dont change, excludes files in the out meaning the netcdf files
excludes=()
#output name of the zip file
groupname="group1.zip"
cwd=$(pwd)

cd ${dirstozip}
for i in * #pond-*
do
    adds+=( "${dirstozip}/${i}" )
    excludes+=( "${dirstozip}/${i}/out/*" )
    rm ${dirstozip}/${i}/out/*
done
cd ${cwd}
addstr=""
excludestr=""

for a in ${adds[@]}
do
    addstr+=" $a"
done

for e in ${excludes[@]}
do
    excludestr+=" $e"
done

echo "zip -r ${groupname} ${dirstozip} -x${excludestr}"
zip -r ${groupname} ${dirstozip}
