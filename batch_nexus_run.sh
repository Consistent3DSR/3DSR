#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --account=nexus
#SBATCH --qos=medium
#SBATCH --mem=48gb  
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH -o ./sbatch_file/%j_%x.out # STDOUT


source activate msss
# set -x
module unload cuda/10.2.89
module add cuda/11.8.0 #gcc/7.5.0

cd /fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine
# sh run_massive_exp.sh
# sh exp_synthetic.sh
# sh run_nerf_synthetic.sh 
# sh run_train.sh
# sh exp.sh
source activate msss
python scripts/run_mipnerf360_LR.py
# sh run_stablesr_pipeline.sh
# sh run_variation_exp.sh
# sh run_train.sh
# sh run_colmap.sh
# sh render_video.sh
