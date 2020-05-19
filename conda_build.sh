#!/bin/bash

set -e

PKG_NAME=nrel-rev

ARRAY=( 3.7 )

export CONDA_BLD_PATH=~/conda-bld

for i in "${ARRAY[@]}"
do
	conda-build conda.recipe/ --python $i --channel=nrel
done

# # convert package to other platforms
# platforms=( osx-64 linux-64 win-64 )
# find $CONDA_BLD_PATH/ -name $PKG_NAME*.tar.bz2 | while read file
# do
#     echo $file
#     for platform in "${platforms[@]}"
#     do
#        conda convert --platform $platform $file  -o $CONDA_BLD_PATH/
#     done
# done

# # upload packages to conda
# find $CONDA_BLD_PATH/ -name $PKG_NAME*.tar.bz2 | while read file
# do
#     echo $file
#     anaconda upload -u nrel $file
# done

# echo "Building and uploading conda package done!"
# rm -rf $CONDA_BLD_PATH/*