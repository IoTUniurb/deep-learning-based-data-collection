#!/bin/bash
#SBATCH --job-name=simulate-dlbdc
#SBATCH --mail-type=END
#SBATCH --mail-user=l.calisti@campus.uniurb.it

singularity exec -p -B /run/user/$UID  /software/remest_tensorflow_24.01-tf2-py3_gpu.sif python3 /home/l.calisti/notebooks/dlds_paper/simulate_alg.py
