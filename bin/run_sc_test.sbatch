#!/bin/bash
#SBATCH --account=naris
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=debug

source activate rev

python sc_test.py
