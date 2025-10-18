#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --partition=vulcan-ampere
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high
#SBATCH --mem=48gb  
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH -o ./sbatch_file/%j_%x.out # STDOUT
#SBATCH --ntasks=4

source activate msss
# set -x
module unload cuda/10.2.89
module add cuda/11.8.0 gcc/7.5.0

cd /fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine
# sh run_massive_exp.sh
# sh run_train_orig.sh
# python run_colmap.py
# sh exp_2.sh
# sh exp.sh
# sh run_train.sh
# sh run_variation_exp.sh
# sh run_train.sh
# sh run_stablesr_pipeline.sh
# sh run_LR_and_upsample_w_stablesr.sh
# sh render_LR_and_upsample_w_stablesr.sh
# python scripts/run_mipnerf360_stablesr.py
# python scripts/run_mipnerf360_LR.py
# python scripts/run_mipnerf360_orig.py
# sh run_both.sh
# sh run_colmap.sh
source activate /fs/nexus-scratch/ytchen/miniconda3/envs/msss

# sh run_stablesr_pipeline.sh
sh exp.sh

