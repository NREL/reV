#!/bin/bash

set -e

PKG_NAME=nrel-rev

PY_VERSION=( 3.7 )

export CONDA_BLD_PATH=~/conda-bld

for py in "${PY_VERSION[@]}"
do
	conda build conda.recipe/ --python=$py --channel=nrel
done

# upload packages to conda
find $CONDA_BLD_PATH/ -name $PKG_NAME*.tar.bz2 | while read file
do
    echo Uploading $file
    anaconda upload -u nrel $file
done

echo "Building and uploading conda package done!"
rm -rf $CONDA_BLD_PATH/*
ls $CONDA_BLD_PATH