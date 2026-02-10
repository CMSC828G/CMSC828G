#!/bin/bash
#SBATCH -A cmsc828-class
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:30:00

# if using python, activate venv
source /scratch/zt1/project/cmsc828/shared/assignment_collectives_venv/bin/activate

# move to project directory
cd /path/to/my/project

# run testing harness C++
./run_tests.sh harness RS 4

# or run testing harness Python
./run_tests_py.sh AG 4

# NOTE: you may want to redirect results from your tests to a file for easier debugging...
# for example, you could append `&> results.log` to one of the above commands
