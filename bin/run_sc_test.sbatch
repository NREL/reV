#!/bin/bash
#SBATCH --account=naris
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=high

source activate rev

python sc_test.py
