#!/bin/bash

cd /home/mrossol/reV/bin/

squeue -u mrossol -t R,PD -n sc_test_2| grep mrossol || sbatch -J sc_test_2 run_sc_test.sbatch
