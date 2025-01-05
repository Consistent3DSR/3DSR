#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high
#SBATCH --mem=48gb  
#SBATCH --gres=gpu:rtxa6000:1


source activate ms
# set -x
module unload cuda/10.2.89
module add cuda/11.8.0 gcc/7.5.0

cd /fs/nexus-projects/dyn3Dscene/Codes/mip-splatting
sh run_train.sh
# sh run_colmap.sh
