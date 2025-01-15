#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o logs/log_%j.out  
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python preshuffle.py --config shuffle_config.json