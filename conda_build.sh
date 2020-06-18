#!/bin/bash

set -e

export CONDA_BLD_PATH=~/conda-bld

conda build conda.recipe/ --channel=nrel

anaconda upload -u nrel $(conda build conda.recipe/ --channel=nrel --output)

echo "Building and uploading conda package done!"
rm -rf $CONDA_BLD_PATH/*
ls $CONDA_BLD_PATH
