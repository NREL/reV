#!/bin/bash

set -e

PKG_NAME=nrel-rev

PY_VERSION=( 3.7 )

export CONDA_BLD_PATH=~/conda-bld
platforms=( osx-64 win-64 )
for py in "${PY_VERSION[@]}"
do
	conda build conda.recipe/ --python=$py --channel=nrel
    file=$(conda build conda.recipe/ --python=$py --output)
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file -o $CONDA_BLD_PATH/
    done
done

# upload packages to conda
find $CONDA_BLD_PATH/ -name $PKG_NAME*.tar.bz2 | while read file
do
    anaconda upload -u nrel $file
done

echo "Building and uploading conda package done!"
rm -rf $CONDA_BLD_PATH/*
ls $CONDA_BLD_PATH