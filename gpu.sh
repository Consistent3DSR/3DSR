# Interactive job

module add cuda/11.3.1 gcc/7.5.0
# module add cuda/11.7.0 gcc/7.5.0
# module add cuda/10.2.89
# module add cuda/11.8.0 gcc/7.5.0

# srun --pty --ntasks 4 --mem=32gb --qos=medium --gres=gpu:rtxa6000:1 --time=8:00:00 --account=nexus bash
srun --pty --ntasks 4 --mem=32gb --qos=medium --gres=gpu:rtxa5000:1 --time=8:00:00 --account=nexus bash
