#!/bin/bash

cd /home/mrossol/reV/bin/

squeue -u mrossol -t R,PD -n sc_test_1| grep mrossol || sbatch -J sc_test_1 run_sc_test.sbatch
