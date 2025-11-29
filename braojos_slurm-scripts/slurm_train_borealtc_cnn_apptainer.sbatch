#!/bin/bash
#SBATCH -J TerrainCNNHusky                                # Job name
#SBATCH --account=gts-elightsey3                          # Charge account
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:V100:1                                 # GPU resources
#SBATCH --mem-per-gpu=80G                                  # Memory per GPU
#SBATCH -t 8:00:00                                        # Job duration
#SBATCH -q embers                                         # QOS name
#SBATCH -o /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC/braojos_slurm-scripts/reports/Report-TerrainCNNHusky-%j.out  # Output log
#SBATCH --mail-type=ALL                                    # Mail preferences
#SBATCH --mail-user=mgutierrez64@gatech.edu               # Email address

cd /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC
module load cuda/12.1.1
APPTAINER_IMG=borealtc.sif
if [ ! -f "$APPTAINER_IMG" ]; then
    echo "Building Apptainer image from Dockerfile..."
    apptainer build "$APPTAINER_IMG" Dockerfile
fi
export DATASET='husky'

echo "Running CNN training inside Apptainer..."
srun --gres=gpu:1 apptainer exec --nv "$APPTAINER_IMG" python3 cnn_train.py
