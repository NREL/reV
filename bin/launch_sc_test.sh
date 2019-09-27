#!/bin/bash

cd /home/mrossol/reV/bin/

squeue -u mrossol -t R,PD -n sc_test| grep mrossol || sbatch -J sc_test run_sc_test.sbatch
