#!/bin/bash
#SBATCH -J running_inference_mamba                           # Job name
#SBATCH --account=gts-elightsey3                      # Charge account
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1                             # GPU resources
#SBATCH --mem-per-gpu=80G                             # Memory per GPU
#SBATCH -t 8:00:00                                    # Job duration
#SBATCH -qembers                                    # QOS name
#SBATCH -o /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/learning_based_terrain_id/BorealTC/braojos_sbatch/reports/Report-running_inference_mamba-%j.out                              # Output and error log
#SBATCH --mail-type=ALL                               # Mail preferences
#SBATCH --mail-user=mgutierrez64@gatech.edu           # Email address

cd $SLURM_SUBMIT_DIR

module load anaconda3/2023.03
module load cuda/12.6.1

conda activate borealtc

cd /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/learning_based_terrain_id/BorealTC
srun python model_inference.py 

conda deactivate