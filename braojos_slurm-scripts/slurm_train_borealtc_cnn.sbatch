#!/bin/bash
#SBATCH -J TerrainCNNHusky                                # Job name
#SBATCH --account=gts-elightsey3                      # Charge account
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:V100:1                             # GPU resources
#SBATCH --mem-per-gpu=80G                             # Memory per GPU
#SBATCH -t 8:00:00                                    # Job duration
#SBATCH -qembers                                     # QOS name
#SBATCH -o /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC/braojos_slurm-scripts/reports/Report-TerrainCNNHusky-%j.out                              # Output and error log
#SBATCH --mail-type=ALL                               # Mail preferences
#SBATCH --mail-user=mgutierrez64@gatech.edu           # Email address

cd /storage/home/hcoda1/0/mgutierrez64/p-elightsey3-0/terrain_classification/BorealTC
buildah build --layers -t borealtc .
container_id=$(
  podman run --device nvidia.com/gpu=all \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e DATASET='husky' --rm --ipc host \
    --mount type=bind,source=.,target=/code/ \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    -d borealtc python3 cnn_train.py
)

stop_container() {
  podman container stop $container_id
  podman logs $container_id
}

trap stop_container EXIT
echo "Container ID: $container_id"
podman wait $container_id
