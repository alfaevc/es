#!/bin/sh
#
#
#SBATCH --account=free # The account name for the job.
#SBATCH --job-name=parallelES  # The job name.
#SBATCH --exclusive
#SBATCH -N 4 # The number of nodes to use.
#SBATCH --time=6:00:00 # The time the job will take to run.

module load anaconda/3-2021.11
module load mujoco
module load gcc

python -m pip install tensorflow --user
python -m pip install gym --user
python -m pip install pybullet --user
conda install swig --user
#python -m pip install box2d-py --user

pip3 install mujoco-py

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/rigel/home/yz3685/.mujoco/mujoco210/bin

python es_gaus_parallel.py


# End of script
