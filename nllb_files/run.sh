#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-16:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o logs/log_%j.out  
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1
python finetune.py --data americas-nlp --dev_src spa_Latn --dev_tgt ctp_Latn --model_dir models/anlp_chatino