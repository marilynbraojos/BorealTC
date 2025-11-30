#!/bin/bash
#SBATCH -J terrain_CNNHUSKY                                                                                                                         # Job name
#SBATCH --account=gts-elightsey3                                                                                                                    # Charge account
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1                                                                                                                           # GPU resources
#SBATCH --mem-per-gpu=80G                                                                                                                           # Memory per GPU
#SBATCH -t 8:00:00                                                                                                                                  # Job duration
#SBATCH -qembers                                                                                                                                    # QOS name
#SBATCH -o /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC/slurm-scripts/reports/terrain_CNNHUSKY-%j.out         # Output and error log
#SBATCH --mail-type=ALL                                                                                                                             # Mail preferences
#SBATCH --mail-user=mgutierrez64@gatech.edu                                                                                                         # Email address

cd $SLURM_SUBMIT_DIR
source /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC/.venv/bin/activate
srun python /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC/cnn_train.py