#!/bin/sh
  #SBATCH -c 1                # Request 1 CPU core
  #SBATCH -t 0-02:00          # Runtime in D-HH:MM, minimum of 10 mins
                                (this requests 2 hours)
  #SBATCH -p dl               # Partition to submit to 
                                (should always be dl, for now)
  #SBATCH --mem=10G           # Request 10G of memory
  #SBATCH -o myoutput_%j.out  # File to which STDOUT will be written
                                (%j inserts jobid)
  #SBATCH -e myerrors_%j.err  # File to which STDERR will be written
                                (%j inserts jobid)
  #SBATCH --gres=gpu:2        # Request two GPUs
  python train.py               # Command you want to run on the cluster
